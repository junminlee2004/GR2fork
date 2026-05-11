// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/config.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"

#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

#include <miniz.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>

namespace {

// Single-mutex producer-consumer queue for cache I/O.
//
// FIX(GR2FORK): the original implementation used two mutexes (`submit_mutex`
// guarding a `num_requests` counter, `m_request` guarding `req_queue`) which
// could drift apart: count > queue.size() was reachable, and the worker's
// inner `while (num_requests) { ... if queue.empty() continue; ... }` would
// spin forever in that state. This rewrite drops the counter entirely — the
// queue itself is the source of truth — and uses one mutex for both the
// queue and the wakeup predicate.
//
// Also drops the redundant `request.get_future().wait()` after running the
// task: `std::packaged_task::operator()` runs the underlying callable
// synchronously and stores the result; the future was never retrieved, so
// the wait was always a no-op (and a future-already-consumed UB risk on
// some implementations).
std::mutex io_queue_mutex{};
std::condition_variable_any io_queue_cv{};
std::queue<std::packaged_task<void()>> io_queue{};

mz_zip_archive zip_ar{};
bool ar_is_read_only{true};

} // namespace

namespace Storage {

void ProcessIO(const std::stop_token& stoken) {
    Common::SetCurrentThreadName("shadPS4:PipelineCacheIO");

    while (!stoken.stop_requested()) {
        std::packaged_task<void()> request{};
        {
            std::unique_lock lk{io_queue_mutex};
            Common::CondvarWait(io_queue_cv, lk, stoken,
                                [&] { return !io_queue.empty(); });
            if (stoken.stop_requested()) {
                break;
            }
            // Predicate guarantees non-empty queue here.
            request = std::move(io_queue.front());
            io_queue.pop();
        }
        if (request.valid()) {
            request();
        }
    }
}

constexpr std::string GetBlobFileExtension(BlobType type) {
    switch (type) {
    case BlobType::ShaderMeta: {
        return "meta";
    }
    case BlobType::ShaderBinary: {
        return "spv";
    }
    case BlobType::PipelineKey: {
        return "key";
    }
    case BlobType::ShaderProfile: {
        return "bin";
    }
    default:
        UNREACHABLE();
    }
}

void DataBase::Open() {
    if (opened) {
        return;
    }

    const auto& game_info = Common::ElfInfo::Instance();

    using namespace Common::FS;
    if (Config::isPipelineCacheArchived()) {
        mz_zip_zero_struct(&zip_ar);

        cache_path = GetUserPath(PathType::CacheDir) /
                     std::filesystem::path{game_info.GameSerial()}.replace_extension(".zip");

        if (!mz_zip_reader_init_file(&zip_ar, cache_path.string().c_str(),
                                     MZ_ZIP_FLAG_READ_ALLOW_WRITING) ||
            !mz_zip_validate_archive(&zip_ar, 0)) {
            LOG_INFO(Render, "Cache archive {} is not found or archive is corrupted",
                     cache_path.string().c_str());
            mz_zip_reader_end(&zip_ar);
            mz_zip_writer_init_file(&zip_ar, cache_path.string().c_str(), 0);
        }
    } else {
        cache_path = GetUserPath(PathType::CacheDir) / game_info.GameSerial();
        if (!std::filesystem::exists(cache_path)) {
            std::filesystem::create_directories(cache_path);
        }
    }

    io_worker = std::jthread{ProcessIO};
    opened = true;
}

void DataBase::Close() {
    if (!IsOpened()) {
        return;
    }

    // FIX(GR2FORK): drain pending writes BEFORE requesting stop. The worker
    // exits on stop_requested without draining; previously any writes still
    // in flight at shutdown were silently lost. Spin-wait briefly with a
    // short sleep — the queue is small (one entry per shader/pipeline blob)
    // and drains in milliseconds. cv.notify_one ensures the worker wakes
    // and processes each entry.
    while (true) {
        bool empty;
        {
            std::scoped_lock lk{io_queue_mutex};
            empty = io_queue.empty();
        }
        if (empty) break;
        io_queue_cv.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    io_worker.request_stop();
    // Worker is currently inside CondvarWait waiting on a non-empty queue —
    // request_stop flips the stop_token but the wait predicate is still
    // !empty. Wake it via notify_one so it observes the stop request.
    io_queue_cv.notify_all();
    io_worker.join();

    if (Config::isPipelineCacheArchived()) {
        mz_zip_writer_finalize_archive(&zip_ar);
        mz_zip_writer_end(&zip_ar);
    }

    opened = false;

    LOG_INFO(Render, "Cache dumped");
}

template <typename T>
bool WriteVector(const BlobType type, std::filesystem::path&& path_, std::vector<T>&& v) {
    auto request = std::packaged_task<void()>{[=]() {
        auto path{path_};
        path.replace_extension(GetBlobFileExtension(type));
        if (Config::isPipelineCacheArchived()) {
            ASSERT_MSG(!ar_is_read_only,
                       "The archive is read-only. Did you forget to call `FinishPreload`?");
            if (!mz_zip_writer_add_mem(&zip_ar, path.string().c_str(), v.data(),
                                       v.size() * sizeof(T), MZ_BEST_COMPRESSION)) {
                LOG_ERROR(Render, "Failed to add {} to the archive", path.string().c_str());
            }
        } else {
            using namespace Common::FS;
            const auto file = IOFile{path, FileAccessMode::Create};
            file.Write(v);
        }
    }};
    {
        std::scoped_lock lk{io_queue_mutex};
        io_queue.emplace(std::move(request));
    }
    io_queue_cv.notify_one();
    return true;
}

template <typename T>
void LoadVector(BlobType type, std::filesystem::path& path, std::vector<T>& v) {
    using namespace Common::FS;
    path.replace_extension(GetBlobFileExtension(type));
    if (Config::isPipelineCacheArchived()) {
        int index{-1};
        index = mz_zip_reader_locate_file(&zip_ar, path.string().c_str(), nullptr, 0);
        if (index < 0) {
            LOG_WARNING(Render, "File {} is not found in the archive", path.string().c_str());
            return;
        }
        mz_zip_archive_file_stat stat{};
        mz_zip_reader_file_stat(&zip_ar, index, &stat);
        v.resize(stat.m_uncomp_size / sizeof(T));
        mz_zip_reader_extract_to_mem(&zip_ar, index, v.data(), stat.m_uncomp_size, 0);
    } else {
        const auto file = IOFile{path, FileAccessMode::Read};
        v.resize(file.GetSize() / sizeof(T));
        file.Read(v);
    }
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u8>&& data) {
    if (!opened) {
        return false;
    }

    auto path = Config::isPipelineCacheArchived() ? std::filesystem::path{name} : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data));
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u32>&& data) {
    if (!opened) {
        return false;
    }

    auto path = Config::isPipelineCacheArchived() ? std::filesystem::path{name} : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data));
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u8>& data) {
    if (!opened) {
        return;
    }

    auto path = Config::isPipelineCacheArchived() ? std::filesystem::path{name} : cache_path / name;
    return LoadVector(type, path, data);
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u32>& data) {
    if (!opened) {
        return;
    }

    auto path = Config::isPipelineCacheArchived() ? std::filesystem::path{name} : cache_path / name;
    return LoadVector(type, path, data);
}

void DataBase::ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func) {
    const auto& ext = GetBlobFileExtension(type);
    if (Config::isPipelineCacheArchived()) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            file_name.fill(0);
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            if (std::string{file_name.data()}.ends_with(ext)) {
                mz_zip_archive_file_stat stat{};
                mz_zip_reader_file_stat(&zip_ar, index, &stat);
                std::vector<u8> data(stat.m_uncomp_size);
                mz_zip_reader_extract_to_mem(&zip_ar, index, data.data(), data.size(), 0);
                func(std::move(data));
            }
        }
    } else {
        for (const auto& file_name : std::filesystem::directory_iterator{cache_path}) {
            if (file_name.path().extension().string().ends_with(ext)) {
                using namespace Common::FS;
                const auto& file = IOFile{file_name, FileAccessMode::Read};
                if (file.IsOpen()) {
                    std::vector<u8> data(file.GetSize());
                    file.Read(data);
                    func(std::move(data));
                }
            }
        }
    }
}

void DataBase::FinishPreload() {
    if (Config::isPipelineCacheArchived()) {
        mz_zip_writer_init_from_reader(&zip_ar, cache_path.string().c_str());
        ar_is_read_only = false;
    }
}

u32 DataBase::CountBlobs(BlobType type) {
    if (!opened) {
        return 0;
    }
    // Counts blobs of `type` without reading their data — directory metadata
    // only (filesystem mode) or zip-index walk (archived mode). Used to size
    // the "LOADING SHADERS" progress bar before the slow ForEachBlob pass
    // that actually deserializes each pipeline. For 41,896-pipeline caches
    // this is a sub-100ms scan on any reasonable filesystem, well below the
    // user-visible flash threshold.
    const auto& ext = GetBlobFileExtension(type);
    u32 count = 0;
    if (Config::isPipelineCacheArchived()) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            file_name.fill(0);
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            if (std::string{file_name.data()}.ends_with(ext)) {
                ++count;
            }
        }
    } else {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator{cache_path, ec}) {
            if (ec) {
                break;
            }
            if (entry.path().extension().string().ends_with(ext)) {
                ++count;
            }
        }
    }
    return count;
}

} // namespace Storage
