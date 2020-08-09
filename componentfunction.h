#ifndef COMPONENTFUNCTION_H_
#define COMPONENTFUNCTION_H_

#include <array>
#include <memory>

#include "instruction.h"

namespace automl_zero {

    class ComponentFunction {
    public:
        std::vector<std::shared_ptr<const Instruction>> instructions;
    };
}

#endif
