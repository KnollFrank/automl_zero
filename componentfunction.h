#ifndef COMPONENTFUNCTION_H_
#define COMPONENTFUNCTION_H_

#include <array>
#include <memory>

#include "instruction.h"

namespace automl_zero
{

    class ComponentFunction
    {
    public:
        bool empty() const;
        int size() const;
        void insertRandomly(RandomGenerator &rand_gen, std::shared_ptr<Instruction> instruction);
        void remove(const InstructionIndexT position);
        bool operator==(const ComponentFunction &other) const;
        bool operator!=(const ComponentFunction &other) const
        {
            return !(*this == other);
        }
        void ShallowCopyTo(ComponentFunction &dest) const;
        std::vector<std::shared_ptr<Instruction>> &getInstructions();
        const std::vector<std::shared_ptr<Instruction>> &getConstInstructions() const;

    private:
        InstructionIndexT RandomInstructionIndex(RandomGenerator &rand_gen, const InstructionIndexT numInstructions);
        std::vector<std::shared_ptr<Instruction>> instructions;
    };
} // namespace automl_zero

#endif
