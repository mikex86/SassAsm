#pragma once

#define REG_RZ 0xFF

struct SassInstructionData {
};

struct SassInstruction {
    virtual void serialize(SassInstructionData &dst);
    virtual ~SassInstruction();
};
