// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>

#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/np/gr2_online_auth.h"
#include "core/libraries/np/np_error.h"
#include "core/libraries/np/np_manager.h"
#include "core/tls.h"

#include "common/singleton.h"
#include "core/linker.h"

namespace Libraries::Np::NpManager {

static bool g_signed_in = false;
static s32 g_active_requests = 0;
static std::mutex g_request_mutex;

// Internal types for storing request-related information
enum class NpRequestState {
    None = 0,
    Ready = 1,
    Aborted = 2,
    Complete = 3,
};

struct NpRequest {
    NpRequestState state;
    bool async;
    s32 result;
};

static std::vector<NpRequest> g_requests;

s32 CreateNpRequest(bool async) {
    if (g_active_requests == ORBIS_NP_MANAGER_REQUEST_LIMIT) {
        return ORBIS_NP_ERROR_REQUEST_MAX;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = 0;
    while (req_index < g_requests.size()) {
        // Find first nonexistant request
        if (g_requests[req_index].state == NpRequestState::None) {
            // There is no request at this index, set the index to ready then break.
            g_requests[req_index].state = NpRequestState::Ready;
            g_requests[req_index].async = async;
            break;
        }
        req_index++;
    }

    if (req_index == g_requests.size()) {
        // There are no requests to replace.
        NpRequest new_request{NpRequestState::Ready, async, 0};
        g_requests.emplace_back(new_request);
    }

    // Offset by one, first returned ID is 0x20000001
    g_active_requests++;
    return req_index + ORBIS_NP_MANAGER_REQUEST_ID_OFFSET + 1;
}

s32 PS4_SYSV_ABI sceNpCreateRequest() {
    LOG_DEBUG(Lib_NpManager, "called");
    return CreateNpRequest(false);
}

s32 PS4_SYSV_ABI sceNpCreateAsyncRequest(const OrbisNpCreateAsyncRequestParameter* param) {
    LOG_DEBUG(Lib_NpManager, "called");
    if (param == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    if (param->size != sizeof(OrbisNpCreateAsyncRequestParameter)) {
        return ORBIS_NP_ERROR_INVALID_SIZE;
    }

    return CreateNpRequest(true);
}

s32 PS4_SYSV_ABI sceNpCheckNpAvailability(s32 req_id, OrbisNpOnlineId* online_id) {
    if (online_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, is_async = {}", req_id,
              request.async);

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpCheckNpAvailabilityA(s32 req_id,
                                           Libraries::UserService::OrbisUserServiceUserId user_id) {
    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, is_async = {}", req_id,
              request.async);

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpCheckNpReachability(s32 req_id,
                                          Libraries::UserService::OrbisUserServiceUserId user_id) {
    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, is_async = {}", req_id,
              request.async);

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpCheckPlus(s32 req_id, const OrbisNpCheckPlusParameter* param,
                                OrbisNpCheckPlusResult* result) {

    if (req_id == 0 || param == nullptr || result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    if (param->size != sizeof(OrbisNpCheckPlusParameter)) {
        return ORBIS_NP_ERROR_INVALID_SIZE;
    }

    if (param->features < 1 || param->features > 3) {
        // TODO: If compiled SDK version is greater or equal to fw 3.50,
        // error if param->features != 1 instead.
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager,
              "(STUBBED) called, req_id = {:#x}, is_async = {}, param.features = {:#x}", req_id,
              request.async, param->features);

    // For now, set authorized to true to signal PS+ access.
    result->authorized = true;

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountLanguage(s32 req_id, OrbisNpOnlineId* online_id,
                                         OrbisNpLanguageCode* language) {
    if (online_id == nullptr || language == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, is_async = {}", req_id,
              request.async);

    std::memset(language, 0, sizeof(OrbisNpLanguageCode));

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountLanguageA(s32 req_id,
                                          Libraries::UserService::OrbisUserServiceUserId user_id,
                                          OrbisNpLanguageCode* language) {
    if (language == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, user_id = {}, is_async = {}",
              req_id, user_id, request.async);

    std::memset(language, 0, sizeof(OrbisNpLanguageCode));

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetParentalControlInfo(s32 req_id, OrbisNpOnlineId* online_id, s8* age,
                                             OrbisNpParentalControlInfo* info) {
    if (online_id == nullptr || age == nullptr || info == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, is_async = {}", req_id,
              request.async);

    // TODO: Add to config?
    *age = 13;
    std::memset(info, 0, sizeof(OrbisNpParentalControlInfo));

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI
sceNpGetParentalControlInfoA(s32 req_id, Libraries::UserService::OrbisUserServiceUserId user_id,
                             s8* age, OrbisNpParentalControlInfo* info) {
    if (age == nullptr || info == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_requests[req_index];
    if (request.state == NpRequestState::Complete) {
        request.result = ORBIS_NP_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpRequestState::Aborted) {
        request.result = ORBIS_NP_ERROR_ABORTED;
        return ORBIS_NP_ERROR_ABORTED;
    }

    request.state = NpRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpManager, "(STUBBED) called, req_id = {:#x}, user_id = {}, is_async = {}",
              req_id, user_id, request.async);

    // TODO: Add to config?
    *age = 13;
    std::memset(info, 0, sizeof(OrbisNpParentalControlInfo));

    request.result = ORBIS_OK;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAbortRequest(s32 req_id) {
    LOG_DEBUG(Lib_NpManager, "called req_id = {:#x}", req_id);

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    if (g_requests[req_index].state == NpRequestState::Complete) {
        // If the request is already complete, abort is ignored.
        return ORBIS_OK;
    }

    g_requests[req_index].state = NpRequestState::Aborted;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWaitAsync(s32 req_id, s32* result) {
    if (result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    if (!g_requests[req_index].async || g_requests[req_index].state == NpRequestState::Ready) {
        return ORBIS_NP_ERROR_INVALID_ID;
    }

    // Since we're not actually performing any sort of network request here,
    // we can just set result based on the request and return.
    *result = g_requests[req_index].result;
    LOG_WARNING(Lib_NpManager, "called req_id = {:#x}, returning result = {:#x}", req_id,
                static_cast<u32>(*result));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpPollAsync(s32 req_id, s32* result) {
    if (result == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    if (!g_requests[req_index].async || g_requests[req_index].state == NpRequestState::Ready) {
        return ORBIS_NP_ERROR_INVALID_ID;
    }

    // Since we're not actually performing any sort of network request here,
    // we can just set result based on the request and return.
    *result = g_requests[req_index].result;
    LOG_WARNING(Lib_NpManager, "called req_id = {:#x}, returning result = {:#x}", req_id,
                static_cast<u32>(*result));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpDeleteRequest(s32 req_id) {
    LOG_DEBUG(Lib_NpManager, "called req_id = {:#x}", req_id);

    std::scoped_lock lk{g_request_mutex};

    s32 req_index = req_id - ORBIS_NP_MANAGER_REQUEST_ID_OFFSET - 1;
    if (g_active_requests == 0 || g_requests.size() <= req_index ||
        g_requests[req_index].state == NpRequestState::None) {
        return ORBIS_NP_ERROR_REQUEST_NOT_FOUND;
    }

    g_active_requests--;
    g_requests[req_index].state = NpRequestState::None;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountCountry(OrbisNpOnlineId* online_id,
                                        OrbisNpCountryCode* country_code) {
    if (online_id == nullptr || country_code == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    std::memset(country_code, 0, sizeof(OrbisNpCountryCode));
    // TODO: get NP country code from config
    std::memcpy(country_code->country_code, "us", 2);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountCountryA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                         OrbisNpCountryCode* country_code) {
    if (country_code == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    std::memset(country_code, 0, sizeof(OrbisNpCountryCode));
    // TODO: get NP country code from config
    std::memcpy(country_code->country_code, "us", 2);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountDateOfBirth(OrbisNpOnlineId* online_id,
                                            OrbisNpDate* date_of_birth) {
    if (online_id == nullptr || date_of_birth == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    // TODO: maybe add to config?
    date_of_birth->day = 1;
    date_of_birth->month = 1;
    date_of_birth->year = 2000;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountDateOfBirthA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                             OrbisNpDate* date_of_birth) {
    if (date_of_birth == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    // TODO: maybe add to config?
    date_of_birth->day = 1;
    date_of_birth->month = 1;
    date_of_birth->year = 2000;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetGamePresenceStatus(OrbisNpOnlineId* online_id,
                                            OrbisNpGamePresenseStatus* game_status) {
    if (online_id == nullptr || game_status == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    *game_status =
        g_signed_in ? OrbisNpGamePresenseStatus::Online : OrbisNpGamePresenseStatus::Offline;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetGamePresenceStatusA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                             OrbisNpGamePresenseStatus* game_status) {
    if (game_status == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    *game_status =
        g_signed_in ? OrbisNpGamePresenseStatus::Online : OrbisNpGamePresenseStatus::Offline;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountId(OrbisNpOnlineId* online_id, u64* account_id) {
    LOG_DEBUG(Lib_NpManager, "GR2: sceNpGetAccountId called");
    if (online_id == nullptr || account_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        *account_id = 0;
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    *account_id = 0xFEEDFACE;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetAccountIdA(Libraries::UserService::OrbisUserServiceUserId user_id,
                                    u64* account_id) {
    LOG_DEBUG(Lib_NpManager, "user_id {}", user_id);
    if (account_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        *account_id = 0;
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    *account_id = 0xFEEDFACE;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetNpId(Libraries::UserService::OrbisUserServiceUserId user_id,
                              OrbisNpId* np_id) {
    LOG_DEBUG(Lib_NpManager, "user_id {}", user_id);
    if (np_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    memset(np_id, 0, sizeof(OrbisNpId));
    strncpy(np_id->handle.data, GR2Fork::Auth::EffectiveOnlineId().c_str(),
            sizeof(np_id->handle.data));
    LOG_DEBUG(Lib_NpManager, "GR2: sceNpGetNpId user_id = {} -> handle = '{}'", user_id,
             np_id->handle.data);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetOnlineId(Libraries::UserService::OrbisUserServiceUserId user_id,
                                  OrbisNpOnlineId* online_id) {
    LOG_DEBUG(Lib_NpManager, "user_id {}", user_id);
    if (online_id == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // GR2FORK: Gravity Rush 2's user-object constructor calls this at user-service login, before PSN
    // sign-in completes, and reads online_id->data even when the call returns SIGNED_OUT, then uses it
    // verbatim as the online-ghost nameplate. The buffer is filled before the sign-in gate so the name
    // is always valid; the return code is unchanged, so the sign-in state machine still sees SIGNED_OUT.
    memset(online_id, 0, sizeof(OrbisNpOnlineId));
    strncpy(online_id->data, GR2Fork::Auth::EffectiveOnlineId().c_str(), sizeof(online_id->data));
    if (!g_signed_in) {
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }
    LOG_DEBUG(Lib_NpManager, "GR2: sceNpGetOnlineId user_id = {} -> online_id = '{}'", user_id,
             online_id->data);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetNpReachabilityState(Libraries::UserService::OrbisUserServiceUserId user_id,
                                             OrbisNpReachabilityState* state) {
    if (state == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }

    *state =
        g_signed_in ? OrbisNpReachabilityState::Reachable : OrbisNpReachabilityState::Unavailable;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpGetState(Libraries::UserService::OrbisUserServiceUserId user_id,
                               OrbisNpState* state) {
    if (state == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    *state = g_signed_in ? OrbisNpState::SignedIn : OrbisNpState::SignedOut;
    LOG_DEBUG(Lib_NpManager, "GR2: sceNpGetState user_id = {} -> Signed {}", user_id,
             g_signed_in ? "in" : "out");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpHasSignedUp(Libraries::UserService::OrbisUserServiceUserId user_id,
                                  bool* has_signed_up) {
    LOG_DEBUG(Lib_NpManager, "called");
    if (has_signed_up == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    *has_signed_up = g_signed_in ? true : false;
    return ORBIS_OK;
}

struct NpStateCallbackForNpToolkit {
    OrbisNpStateCallbackForNpToolkit func;
    void* userdata;
};

NpStateCallbackForNpToolkit NpStateCbForNp;

// GR2 online restoration: libSceNpToolkit.prx registers a callback via
// sceNpRegisterStateCallbackForToolkit and waits for sign-in notification, never polling - an
// undelivered state stalls it forever. Delivered once per change from sceNpCheckCallback[ForLib].
static bool g_np_toolkit_state_dirty = false;

// GR2: the toolkit also registers an NP reachability callback
// (sceNpRegisterNpReachabilityStateCallback) and waits to be told it is network-reachable;
// SignedIn alone does not unblock it. Same delivery model; signature (userId, state, userdata).
using OrbisNpReachabilityStateCallback =
    PS4_SYSV_ABI void (*)(s32 userId, OrbisNpReachabilityState state, void* userdata);

struct NpReachabilityStateCallback {
    OrbisNpReachabilityStateCallback func;
    void* userdata;
};

NpReachabilityStateCallback NpReachabilityCb;
static bool g_np_reachability_dirty = false;

static void DeliverNpToolkitState() {
    s32 user_id = 0;
    bool have_user = false;
    const auto ensure_user = [&] {
        if (!have_user) {
            Libraries::UserService::sceUserServiceGetInitialUser(&user_id);
            have_user = true;
        }
    };

    // Deliver sign-in state, once per change. PS4 delivers NP callbacks from this pump.
    if (NpStateCbForNp.func != nullptr && g_np_toolkit_state_dirty) {
        // Clear before invoking so a re-entrant / multi-threaded pump cannot double-deliver.
        g_np_toolkit_state_dirty = false;
        ensure_user();
        const OrbisNpState state = g_signed_in ? OrbisNpState::SignedIn : OrbisNpState::SignedOut;
        LOG_INFO(Lib_NpManager, "GR2: delivering NP state {} to toolkit callback, user_id = {}",
                 state == OrbisNpState::SignedIn ? "SignedIn" : "SignedOut", user_id);
        // Invoke the guest callback through the host->guest helper so the guest TLS/stack are set
        // up correctly (this is why core/tls.h is included).
        Core::ExecuteGuest(NpStateCbForNp.func, user_id, state, NpStateCbForNp.userdata);
    }

    // Deliver reachability state, once per change.
    if (NpReachabilityCb.func != nullptr && g_np_reachability_dirty) {
        g_np_reachability_dirty = false;
        ensure_user();
        const OrbisNpReachabilityState rstate = g_signed_in ? OrbisNpReachabilityState::Reachable
                                                            : OrbisNpReachabilityState::Unavailable;
        LOG_INFO(Lib_NpManager,
                 "GR2: delivering NP reachability {} to toolkit callback, user_id = {}",
                 rstate == OrbisNpReachabilityState::Reachable ? "Reachable" : "Unavailable",
                 user_id);
        Core::ExecuteGuest(NpReachabilityCb.func, user_id, rstate, NpReachabilityCb.userdata);
    }
}

s32 PS4_SYSV_ABI sceNpCheckCallback() {
    DeliverNpToolkitState();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpCheckCallbackForLib() {
    DeliverNpToolkitState();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpRegisterStateCallbackForToolkit(OrbisNpStateCallbackForNpToolkit callback,
                                                      void* userdata) {
    static s32 id = 0;
    NpStateCbForNp.func = callback;
    NpStateCbForNp.userdata = userdata;
    // Deliver the current sign-in state on the next callback pump.
    g_np_toolkit_state_dirty = true;
    LOG_INFO(Lib_NpManager, "GR2: toolkit state callback registered, signed_in = {}", g_signed_in);
    return id;
}

s32 PS4_SYSV_ABI sceNpRegisterNpReachabilityStateCallback(
    OrbisNpReachabilityStateCallback callback, void* userdata) {
    NpReachabilityCb.func = callback;
    NpReachabilityCb.userdata = userdata;
    // Deliver the current reachability state on the next callback pump.
    g_np_reachability_dirty = true;
    LOG_INFO(Lib_NpManager, "GR2: reachability callback registered, signed_in = {}", g_signed_in);
    return ORBIS_OK;
}

// GR2FORK FIX: NpToolkit resolves each in-race rank-list row and overworld ghost-nameplate avatar by
// calling UserProfile::Interface::getAvatarUrl(Future<string>* out, request, async), then GETs the
// returned URL and draws it into the icon slot (a placeholder Kat when the URL is empty). The bundled
// PRX's getAvatarUrl never yields a URL on the emulator, so every icon stays the placeholder. HLE
// symbols resolve ahead of a loaded PRX, so this shadow is called instead and hands back a ready Future
// holding the restoration server's per-onlineId avatar URL; the game's own loader fetches and decodes
// it (ReplaceHost reroutes gr2.local to the server). Future<string> PRX ABI: state s32 @ out+0x18
// (0 = done+success); result std::string @ out+0x230 (data @ +0x08, size @ +0x18, capacity @ +0x20;
// capacity > 0xf means the data union holds a heap pointer, which the PRX dtor frees through the
// allocator object at PRX_base+0xb0258 -- so a long URL must be allocated through that same allocator).
static u64 Gr2NpToolkitModuleBase() {
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    if (linker == nullptr) {
        return 0;
    }
    for (s32 i = 0; i < 512; ++i) {
        auto* m = linker->GetModule(i);
        if (m == nullptr) {
            break;
        }
        if (m->name.contains("libSceNpToolkit")) {
            return m->GetBaseAddress();
        }
    }
    return 0;
}

// A PSN Online ID is printable ASCII, at most 16 chars. Distinguishes a real target Online ID
// from the self-request form, where request+0x30 is null or an uninitialized non-string.
static bool IsPlausibleOnlineId(const char* s) {
    if (s == nullptr) {
        return false;
    }
    for (int i = 0; i <= 16; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\0') {
            return i > 0;
        }
        if (c < 0x21 || c > 0x7e) {
            return false;
        }
    }
    return false;
}

s32 PS4_SYSV_ABI sceToolkitUserProfileGetAvatarUrl(u64 future, const u8* request, bool async) {
    if (future == 0) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // request+0x30 holds the target Online ID on a targeted request; on the self-request form it is
    // null or an uninitialized non-string. A non-plausible Online ID resolves to the
    // authenticated user, so the profile card shows the logged-in avatar, not a garbage lookup.
    const char* oid = request ? *reinterpret_cast<const char* const*>(request + 0x30) : nullptr;
    char who[17] = {0};
    if (IsPlausibleOnlineId(oid)) {
        strncpy(who, oid, sizeof(who) - 1);
    } else {
        strncpy(who, GR2Fork::Auth::EffectiveOnlineId().c_str(), sizeof(who) - 1);
    }
    const std::string url = std::string("http://gr2.local:8443/avatar.png?u=") + who;
    const u64 len = url.size();

    auto* out = reinterpret_cast<u8*>(future);
    u8* str = out + 0x230; // the result std::string
    if (len <= 15) {       // SSO (short-string): data is inline
        memcpy(str + 0x08, url.c_str(), len + 1);
        *reinterpret_cast<u64*>(str + 0x18) = len;
        *reinterpret_cast<u64*>(str + 0x20) = 0x0f;
    } else { // heap: allocate through the PRX allocator so its own dtor frees the buffer correctly
        const u64 base = Gr2NpToolkitModuleBase();
        void* alloc_obj = base ? *reinterpret_cast<void**>(base + 0xb0258) : nullptr;
        if (alloc_obj == nullptr) {
            // The Future is left pending, so the consumer draws an EMPTY icon rather than the
            // server's default image. Report it: an icon that is blank instead of the fallback
            // picture is this path, not a missing avatar file or a failed download.
            LOG_ERROR(Lib_NpManager,
                      "GR2: getAvatarUrl('{}') -> NpToolkit allocator unavailable (module base "
                      "{:#x}); Future left pending, icon stays blank",
                      who, base);
            return ORBIS_NP_ERROR_INVALID_ARGUMENT;
        }
        using AllocFn = PS4_SYSV_ABI void* (*)(void* self, u64 size);
        auto** vtable = *reinterpret_cast<void***>(alloc_obj);
        char* buf = reinterpret_cast<char*>(
            Core::ExecuteGuest(reinterpret_cast<AllocFn>(vtable[0]), alloc_obj, len + 1));
        if (buf == nullptr) {
            return ORBIS_NP_ERROR_INVALID_ARGUMENT;
        }
        memcpy(buf, url.c_str(), len + 1);
        *reinterpret_cast<char**>(str + 0x08) = buf;
        *reinterpret_cast<u64*>(str + 0x18) = len;
        *reinterpret_cast<u64*>(str + 0x20) = len; // > 0xf -> consumer dereferences str+0x08
    }
    *reinterpret_cast<volatile s32*>(out + 0x18) = 0; // state = done + success
    LOG_INFO(Lib_NpManager, "GR2: getAvatarUrl('{}') -> {} [oid={} {}]", who, url,
             oid ? oid : "(null)", IsPlausibleOnlineId(oid) ? "targeted" : "self/fallback");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    g_signed_in = Config::getPSNSignedIn();

    LIB_FUNCTION("GpLQDNKICac", "libSceNpManager", 1, "libSceNpManager", sceNpCreateRequest);
    LIB_FUNCTION("eiqMCt9UshI", "libSceNpManager", 1, "libSceNpManager", sceNpCreateAsyncRequest);
    LIB_FUNCTION("2rsFmlGWleQ", "libSceNpManager", 1, "libSceNpManager", sceNpCheckNpAvailability);
    LIB_FUNCTION("8Z2Jc5GvGDI", "libSceNpManager", 1, "libSceNpManager", sceNpCheckNpAvailabilityA);
    LIB_FUNCTION("KfGZg2y73oM", "libSceNpManager", 1, "libSceNpManager", sceNpCheckNpReachability);
    LIB_FUNCTION("r6MyYJkryz8", "libSceNpManager", 1, "libSceNpManager", sceNpCheckPlus);
    LIB_FUNCTION("KZ1Mj9yEGYc", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountLanguage);
    LIB_FUNCTION("TPMbgIxvog0", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountLanguageA);
    LIB_FUNCTION("ilwLM4zOmu4", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetParentalControlInfo);
    LIB_FUNCTION("m9L3O6yst-U", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetParentalControlInfoA);
    LIB_FUNCTION("OzKvTvg3ZYU", "libSceNpManager", 1, "libSceNpManager", sceNpAbortRequest);
    LIB_FUNCTION("jyi5p9XWUSs", "libSceNpManager", 1, "libSceNpManager", sceNpWaitAsync);
    LIB_FUNCTION("uqcPJLWL08M", "libSceNpManager", 1, "libSceNpManager", sceNpPollAsync);
    LIB_FUNCTION("S7QTn72PrDw", "libSceNpManager", 1, "libSceNpManager", sceNpDeleteRequest);

    LIB_FUNCTION("Ghz9iWDUtC4", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountCountry);
    LIB_FUNCTION("JT+t00a3TxA", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountCountryA);
    LIB_FUNCTION("8VBTeRf1ZwI", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetAccountDateOfBirth);
    LIB_FUNCTION("q3M7XzBKC3s", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetAccountDateOfBirthA);
    LIB_FUNCTION("IPb1hd1wAGc", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetGamePresenceStatus);
    LIB_FUNCTION("oPO9U42YpgI", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetGamePresenceStatusA);
    LIB_FUNCTION("e-ZuhGEoeC4", "libSceNpManager", 1, "libSceNpManager",
                 sceNpGetNpReachabilityState);

    LIB_FUNCTION("a8R9-75u4iM", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountId);
    LIB_FUNCTION("rbknaUjpqWo", "libSceNpManager", 1, "libSceNpManager", sceNpGetAccountIdA);
    LIB_FUNCTION("p-o74CnoNzY", "libSceNpManager", 1, "libSceNpManager", sceNpGetNpId);
    LIB_FUNCTION("XDncXQIJUSk", "libSceNpManager", 1, "libSceNpManager", sceNpGetOnlineId);
    LIB_FUNCTION("eQH7nWPcAgc", "libSceNpManager", 1, "libSceNpManager", sceNpGetState);
    LIB_FUNCTION("Oad3rvY-NJQ", "libSceNpManager", 1, "libSceNpManager", sceNpHasSignedUp);
    LIB_FUNCTION("3Zl8BePTh9Y", "libSceNpManager", 1, "libSceNpManager", sceNpCheckCallback);
    LIB_FUNCTION("JELHf4xPufo", "libSceNpManager", 1, "libSceNpManager", sceNpCheckCallbackForLib);
    LIB_FUNCTION("JELHf4xPufo", "libSceNpManagerForToolkit", 1, "libSceNpManager",
                 sceNpCheckCallbackForLib);
    LIB_FUNCTION("0c7HbXRKUt4", "libSceNpManagerForToolkit", 1, "libSceNpManager",
                 sceNpRegisterStateCallbackForToolkit);
    LIB_FUNCTION("hw5KNqAAels", "libSceNpManager", 1, "libSceNpManager",
                 sceNpRegisterNpReachabilityStateCallback);

    // GR2FORK: shadow NpToolkit UserProfile::getAvatarUrl (3-arg form the game imports) so the in-race
    // rank-list rows and ghost nameplates resolve a per-player avatar URL instead of the placeholder.
    LIB_FUNCTION("3jeBu0QFXWU", "libSceNpToolkit", 1, "libSceNpToolkit",
                 sceToolkitUserProfileGetAvatarUrl);

    LIB_FUNCTION("2rsFmlGWleQ", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpCheckNpAvailability);
    LIB_FUNCTION("a8R9-75u4iM", "libSceNpManagerCompat", 1, "libSceNpManager", sceNpGetAccountId);
    LIB_FUNCTION("KZ1Mj9yEGYc", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetAccountLanguage);
    LIB_FUNCTION("Ghz9iWDUtC4", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetAccountCountry);
    LIB_FUNCTION("8VBTeRf1ZwI", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetAccountDateOfBirth);
    LIB_FUNCTION("IPb1hd1wAGc", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetGamePresenceStatus);
    LIB_FUNCTION("ilwLM4zOmu4", "libSceNpManagerCompat", 1, "libSceNpManager",
                 sceNpGetParentalControlInfo);
};

} // namespace Libraries::Np::NpManager
