// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "componentfunction.h"

#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace automl_zero
{

  using ::std::vector; // NOLINT
  using ::testing::ElementsAre;

  TEST(ComponentfunctionTest, SizeOfEmptyComponentFunctionIs0)
  {
    // Given
    ComponentFunction componentFunction;

    // When
    int size = componentFunction.size();

    // Then
    EXPECT_EQ(size, 0);
  }

  TEST(ComponentfunctionTest, SizeOfSingleInstructionComponentFunctionIs1)
  {
    // Given
    ComponentFunction componentFunction;
    componentFunction.getInstructions().emplace_back(std::make_shared<Instruction>(
        SCALAR_CONST_SET_OP,
        2,
        ActivationDataSetter(1.0)));

    // When
    int size = componentFunction.size();

    // Then
    EXPECT_EQ(size, 1);
  }

  TEST(ComponentfunctionTest, SizeOfLoopWithEmptyBodyIs1)
  {
    // Given
    ComponentFunction componentFunction;
    Instruction loop = Instruction(LOOP, 5, 6);
    componentFunction.getInstructions().emplace_back(std::make_shared<Instruction>(loop));

    // When
    int size = componentFunction.size();

    // Then
    EXPECT_EQ(size, 1);
  }

  TEST(ComponentfunctionTest, SizeOfLoopWithSingleInstructionInBodyIs2)
  {
    // Given
    ComponentFunction componentFunction;
    Instruction loop = Instruction(LOOP, 5, 6);
    loop.children_.emplace_back(std::make_shared<Instruction>(
        SCALAR_DIFF_OP,
        6,
        2,
        0));
    componentFunction.getInstructions().emplace_back(std::make_shared<Instruction>(loop));

    // When
    int size = componentFunction.size();

    // Then
    EXPECT_EQ(size, 2); // LOOP + SCALAR_DIFF_OP
  }
} // namespace automl_zero
