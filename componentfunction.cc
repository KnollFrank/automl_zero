#include "componentfunction.h"
#include "definitions.h"

namespace automl_zero
{

    bool ComponentFunction::empty() const
    {
        return instructions.empty();
    }

    int ComponentFunction::size() const
    {
        return instructions.size();
    }

    // FK-TODO: DRY with Mutator::RandomInstructionIndex
    InstructionIndexT ComponentFunction::RandomInstructionIndex(RandomGenerator &rand_gen, const InstructionIndexT numInstructions)
    {
        return rand_gen.UniformInteger(0, numInstructions);
    }

    void ComponentFunction::insertRandomly(RandomGenerator &rand_gen, std::shared_ptr<Instruction> instruction)
    {
        const InstructionIndexT position = RandomInstructionIndex(rand_gen, size());
        if (instructions.size() >= 1 && instructions[position]->op_ == LOOP)
        {
            std::vector<std::shared_ptr<Instruction>> &loopInstructions = instructions[position]->children_;
            loopInstructions.insert(
                loopInstructions.begin() + RandomInstructionIndex(rand_gen, loopInstructions.size() + 1),
                instruction);
        }
        else
        {
            instructions.insert(
                instructions.begin() + position,
                instruction);
        }
    }

    void ComponentFunction::remove(const InstructionIndexT position)
    {
        CHECK_GT(instructions.size(), 0);
        instructions.erase(instructions.begin() + position);
    }

    bool ComponentFunction::operator==(const ComponentFunction &other) const
    {
        const std::vector<std::shared_ptr<Instruction>> &component_function1 = this->instructions;
        const std::vector<std::shared_ptr<Instruction>> &component_function2 = other.getConstInstructions();
        if (component_function1.size() != component_function2.size())
        {
            return false;
        }
        std::vector<std::shared_ptr<Instruction>>::const_iterator instruction1_it = component_function1.begin();
        for (const std::shared_ptr<Instruction> &instruction2 : component_function2)
        {
            if (*instruction2 != **instruction1_it)
                return false;
            ++instruction1_it;
        }
        CHECK(instruction1_it == component_function1.end());
        return true;
    }

    void ComponentFunction::ShallowCopyTo(ComponentFunction &dest) const
    {
        dest.getInstructions().reserve(size());
        dest.getInstructions().clear();
        for (const std::shared_ptr<Instruction> &src_instr : instructions)
        {
            dest.getInstructions().emplace_back(src_instr);
        }
    }

    std::vector<std::shared_ptr<Instruction>> &ComponentFunction::getInstructions()
    {
        return instructions;
    }

    const std::vector<std::shared_ptr<Instruction>> &ComponentFunction::getConstInstructions() const
    {
        return instructions;
    }

} // namespace automl_zero