#pragma once

#include <unordered_map>

enum SpecialRegisterExpression
{
    SR_LANE_ID = 0,
    SR_CLOCK = 1,
    SR_VIRTCFG = 2,
    SR_VIRTID = 3,
    SR_ORDERING_TICKET = 15,
    SR_PRIM_TYPE = 16,
    SR_INVOCATION_ID = 17,
    SR_Y_DIRECTION = 18,
    SR_THREAD_KILL = 19,
    SR_SHADER_TYPE = 20,
    SR_DIRECTCBEWRITEADDRESSLOW = 21,
    SR_DIRECTCBEWRITEADDRESSHIGH = 22,
    SR_DIRECTCBEWRITEENABLED = 23,
    SR_SW_SCRATCH = 24,
    SR_MACHINE_ID_1 = 25,
    SR_MACHINE_ID_2 = 26,
    SR_MACHINE_ID_3 = 27,
    SR_AFFINITY = 28,
    SR_INVOCATION_INFO = 29,
    SR_WSCALEFACTOR_XY = 30,
    SR_WSCALEFACTOR_Z = 31,
    SR_TID = 32,
    SR_TID_X = 33,
    SR_TID_Y = 34,
    SR_TID_Z = 35,
    SR_CTAID_X = 37,
    SR_CTAID_Y = 38,
    SR_CTAID_Z = 39,
    SR_NTID = 40,
    SR_CirQueueIncrMinusOne = 41,
    SR_NLATC = 42,
    SR_SM_SPA_VERSION = 44,
    SR_MULTIPASSSHADERINFO = 45,
    SR_LWINHI = 46,
    SR_SWINHI = 47,
    SR_SWINLO = 48,
    SR_SWINSZ = 49,
    SR_SMEMSZ = 50,
    SR_SMEMBANKS = 51,
    SR_LWINLO = 52,
    SR_LWINSZ = 53,
    SR_LMEMLOSZ = 54,
    SR_LMEMHIOFF = 55,
    SR_EQMASK = 56,
    SR_LTMASK = 57,
    SR_LEMASK = 58,
    SR_GTMASK = 59,
    SR_GEMASK = 60,
    SR_REGALLOC = 61,
    SR_BARRIERALLOC = 62,
    SR_GLOBALERRORSTATUS = 64,
    SR_WARPERRORSTATUS = 66,
    SR_VIRTUALSMID = 67,
    SR_VIRTUALENGINEID = 68,
    SR_CLOCKLO = 80,
    SR_CLOCKHI = 81,
    SR_GLOBALTIMERLO = 82,
    SR_GLOBALTIMERHI = 83,
    SR_ESR_PC = 84,
    SR_ESR_PC_HI = 85,
    SR_HWTASKID = 96,
    SR_CIRCULARQUEUEENTRYINDEX = 97,
    SR_CIRCULARQUEUEENTRYADDRESSLOW = 98,
    SR_CIRCULARQUEUEENTRYADDRESSHIGH = 99,
    SR_PM0 = 100,
    SR_PM_HI0 = 101,
    SR_PM1 = 102,
    SR_PM_HI1 = 103,
    SR_PM2 = 104,
    SR_PM_HI2 = 105,
    SR_PM3 = 106,
    SR_PM_HI3 = 107,
    SR_PM4 = 108,
    SR_PM_HI4 = 109,
    SR_PM5 = 110,
    SR_PM_HI5 = 111,
    SR_PM6 = 112,
    SR_PM_HI6 = 113,
    SR_PM7 = 114,
    SR_PM_HI7 = 115,
    SR_SNAP_PM0 = 116,
    SR_SNAP_PM_HI0 = 117,
    SR_SNAP_PM1 = 118,
    SR_SNAP_PM_HI1 = 119,
    SR_SNAP_PM2 = 120,
    SR_SNAP_PM_HI2 = 121,
    SR_SNAP_PM3 = 122,
    SR_SNAP_PM_HI3 = 123,
    SR_SNAP_PM4 = 124,
    SR_SNAP_PM_HI4 = 125,
    SR_SNAP_PM5 = 126,
    SR_SNAP_PM_HI5 = 127,
    SR_SNAP_PM6 = 128,
    SR_SNAP_PM_HI6 = 129,
    SR_SNAP_PM7 = 130,
    SR_SNAP_PM_HI7 = 131,
    SR_VARIABLE_RATE = 132,
    SR_TTU_TICKET_INFO = 133,
    SRZ = 255
};


inline std::unordered_map<std::string, SpecialRegisterExpression> asm_literal_to_sr = {
    {"SR_LANEID", SR_LANE_ID},
    {"SR_CLOCK", SR_CLOCK},
    {"SR_VIRTCFG", SR_VIRTCFG},
    {"SR_VIRTID", SR_VIRTID},
    {"SR_ORDERING_TICKET", SR_ORDERING_TICKET},
    {"SR_PRIM_TYPE", SR_PRIM_TYPE},
    {"SR_INVOCATION_ID", SR_INVOCATION_ID},
    {"SR_Y_DIRECTION", SR_Y_DIRECTION},
    {"SR_THREAD_KILL", SR_THREAD_KILL},
    {"SR_SHADER_TYPE", SR_SHADER_TYPE},
    {"SR_DIRECTCBEWRITEADDRESSLOW", SR_DIRECTCBEWRITEADDRESSLOW},
    {"SR_DIRECTCBEWRITEADDRESSHIGH", SR_DIRECTCBEWRITEADDRESSHIGH},
    {"SR_DIRECTCBEWRITEENABLED", SR_DIRECTCBEWRITEENABLED},
    {"SR_SW_SCRATCH", SR_SW_SCRATCH},
    {"SR_MACHINE_ID_1", SR_MACHINE_ID_1},
    {"SR_MACHINE_ID_2", SR_MACHINE_ID_2},
    {"SR_MACHINE_ID_3", SR_MACHINE_ID_3},
    {"SR_AFFINITY", SR_AFFINITY},
    {"SR_INVOCATION_INFO", SR_INVOCATION_INFO},
    {"SR_WSCALEFACTOR_XY", SR_WSCALEFACTOR_XY},
    {"SR_WSCALEFACTOR_Z", SR_WSCALEFACTOR_Z},
    {"SR_TID", SR_TID},
    {"SR_TID.X", SR_TID_X},
    {"SR_TID.Y", SR_TID_Y},
    {"SR_TID.Z", SR_TID_Z},
    {"SR_CTAID.X", SR_CTAID_X},
    {"SR_CTAID.Y", SR_CTAID_Y},
    {"SR_CTAID.Z", SR_CTAID_Z},
    {"SR_NTID", SR_NTID},
    {"SR_CirQueueIncrMinusOne", SR_CirQueueIncrMinusOne},
    {"SR_NLATC", SR_NLATC},
    {"SR_SM_SPA_VERSION", SR_SM_SPA_VERSION},
    {"SR_MULTIPASSSHADERINFO", SR_MULTIPASSSHADERINFO},
    {"SR_LWINHI", SR_LWINHI},
    {"SR_SWINHI", SR_SWINHI},
    {"SR_SWINLO", SR_SWINLO},
    {"SR_SWINSZ", SR_SWINSZ},
    {"SR_SMEMSZ", SR_SMEMSZ},
    {"SR_SMEMBANKS", SR_SMEMBANKS},
    {"SR_LWINLO", SR_LWINLO},
    {"SR_LWINSZ", SR_LWINSZ},
    {"SR_LMEMLOSZ", SR_LMEMLOSZ},
    {"SR_LMEMHIOFF", SR_LMEMHIOFF},
    {"SR_EQMASK", SR_EQMASK},
    {"SR_LTMASK", SR_LTMASK},
    {"SR_LEMASK", SR_LEMASK},
    {"SR_GTMASK", SR_GTMASK},
    {"SR_GEMASK", SR_GEMASK},
    {"SR_REGALLOC", SR_REGALLOC},
    {"SR_BARRIERALLOC", SR_BARRIERALLOC},
    {"SR_GLOBALERRORSTATUS", SR_GLOBALERRORSTATUS},
    {"SR_WARPERRORSTATUS", SR_WARPERRORSTATUS},
    {"SR_VIRTUALSMID", SR_VIRTUALSMID},
    {"SR_VIRTUALENGINEID", SR_VIRTUALENGINEID},
    {"SR_CLOCKLO", SR_CLOCKLO},
    {"SR_CLOCKHI", SR_CLOCKHI},
    {"SR_GLOBALTIMERLO", SR_GLOBALTIMERLO},
    {"SR_GLOBALTIMERHI", SR_GLOBALTIMERHI},
    {"SR_ESR_PC", SR_ESR_PC},
    {"SR_ESR_PC_HI", SR_ESR_PC_HI},
    {"SR_HWTASKID", SR_HWTASKID},
    {"SR_CIRCULARQUEUEENTRYINDEX", SR_CIRCULARQUEUEENTRYINDEX},
    {"SR_CIRCULARQUEUEENTRYADDRESSLOW", SR_CIRCULARQUEUEENTRYADDRESSLOW},
    {"SR_CIRCULARQUEUEENTRYADDRESSHIGH", SR_CIRCULARQUEUEENTRYADDRESSHIGH},
    {"SR_PM0", SR_PM0},
    {"SR_PM_HI0", SR_PM_HI0},
    {"SR_PM1", SR_PM1},
    {"SR_PM_HI1", SR_PM_HI1},
    {"SR_PM2", SR_PM2},
    {"SR_PM_HI2", SR_PM_HI2},
    {"SR_PM3", SR_PM3},
    {"SR_PM_HI3", SR_PM_HI3},
    {"SR_PM4", SR_PM4},
    {"SR_PM_HI4", SR_PM_HI4},
    {"SR_PM5", SR_PM5},
    {"SR_PM_HI5", SR_PM_HI5},
    {"SR_PM6", SR_PM6},
    {"SR_PM_HI6", SR_PM_HI6},
    {"SR_PM7", SR_PM7},
    {"SR_PM_HI7", SR_PM_HI7},
    {"SR_SNAP_PM0", SR_SNAP_PM0},
    {"SR_SNAP_PM_HI0", SR_SNAP_PM_HI0},
    {"SR_SNAP_PM1", SR_SNAP_PM1},
    {"SR_SNAP_PM_HI1", SR_SNAP_PM_HI1},
    {"SR_SNAP_PM2", SR_SNAP_PM2},
    {"SR_SNAP_PM_HI2", SR_SNAP_PM_HI2},
    {"SR_SNAP_PM3", SR_SNAP_PM3},
    {"SR_SNAP_PM_HI3", SR_SNAP_PM_HI3},
    {"SR_SNAP_PM4", SR_SNAP_PM4},
    {"SR_SNAP_PM_HI4", SR_SNAP_PM_HI4},
    {"SR_SNAP_PM5", SR_SNAP_PM5},
    {"SR_SNAP_PM_HI5", SR_SNAP_PM_HI5},
    {"SR_SNAP_PM6", SR_SNAP_PM6},
    {"SR_SNAP_PM_HI6", SR_SNAP_PM_HI6},
    {"SR_SNAP_PM7", SR_SNAP_PM7},
    {"SR_SNAP_PM_HI7", SR_SNAP_PM_HI7},
    {"SR_VARIABLE_RATE", SR_VARIABLE_RATE},
    {"SR_TTU_TICKET_INFO", SR_TTU_TICKET_INFO},
    {"SRZ", SRZ}
};
