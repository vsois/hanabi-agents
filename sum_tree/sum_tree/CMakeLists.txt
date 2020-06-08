include_directories(include)

add_executable(test_sum_tree src/test_sum_tree.cc)
target_link_libraries(test_sum_tree gtest_main)
add_test(NAME gtest_sum_tree COMMAND test_sum_tree)

# add_library(SumTree src/sum_tree.cc)
pybind11_add_module(SumTree src/sum_tree_py.cc)
target_link_libraries(SumTree PRIVATE OpenMP::OpenMP_CXX)
