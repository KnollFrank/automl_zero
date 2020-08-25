#include "componentfunction.h"
#include "definitions.h"

namespace automl_zero
{

    bool ComponentFunction::empty() const
    {
        return instructions.empty();
    }

    int ComponentFunction::sizeOf(const std::vector<std::shared_ptr<Instruction>> &instructions) const
    {
        int size = 0;
        for (const std::shared_ptr<Instruction> instruction : instructions)
        {
            size += 1 + sizeOf(instruction->children_);
        }
        return size;
    }

    int ComponentFunction::size() const
    {
        return sizeOf(instructions);
    }

    // FK-TODO: DRY with Mutator::RandomInstructionIndex
    InstructionIndexT ComponentFunction::RandomInstructionIndex(RandomGenerator &rand_gen, const InstructionIndexT numInstructions)
    {
        return rand_gen.UniformInteger(0, numInstructions);
    }

    void ComponentFunction::insertRandomly(RandomGenerator &rand_gen, std::shared_ptr<Instruction> instruction)
    {
        const InstructionIndexT position = RandomInstructionIndex(rand_gen, instructions.size() + 1);
        if (position < instructions.size() && instructions[position]->op_ == LOOP)
        {
            switch (rand_gen.Choice2())
            {
            case kChoice0of2:
            {
                // insert instruction into loop body
                std::vector<std::shared_ptr<Instruction>> &loopInstructions = instructions[position]->children_;
                // FK-FIXME: LOOP könnte auch verschachtelt sein, also insertRandomly() irgendwie rekursiv aufrufen
                loopInstructions.insert(
                    loopInstructions.begin() + RandomInstructionIndex(rand_gen, loopInstructions.size() + 1),
                    instruction);
                break;
            }
            case kChoice1of2:
                // insert instruction before loop instruction
                instructions.insert(
                    instructions.begin() + position,
                    instruction);
                break;
            }
        }
        else
        {
            instructions.insert(
                instructions.begin() + position,
                instruction);
        }
    }

    void ComponentFunction::removeRandomly(RandomGenerator &rand_gen)
    {
        CHECK_GT(instructions.size(), 0);
        const InstructionIndexT position = RandomInstructionIndex(rand_gen, instructions.size());
        if (position < instructions.size() && instructions[position]->op_ == LOOP && !instructions[position]->children_.empty())
        {
            switch (rand_gen.Choice2())
            {
            case kChoice0of2:
            {
                // remove some instruction within loop body
                // FK-FIXME: LOOP könnte auch verschachtelt sein, also removeRandomly() irgendwie rekursiv aufrufen
                std::vector<std::shared_ptr<Instruction>> &loopInstructions = instructions[position]->children_;
                loopInstructions.erase(loopInstructions.begin() + RandomInstructionIndex(rand_gen, loopInstructions.size()));
                break;
            }
            case kChoice1of2:
                // remove loop instruction
                instructions.erase(instructions.begin() + position);
                break;
            }
        }
        else
        {
            instructions.erase(instructions.begin() + position);
        }
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