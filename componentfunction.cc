#include "componentfunction.h"

namespace automl_zero {

    bool ComponentFunction::empty() const {
        return instructions.empty();
    }

    int ComponentFunction::size() const {
        return instructions.size();
    }
}