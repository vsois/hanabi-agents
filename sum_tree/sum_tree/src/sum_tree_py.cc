#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sum_tree.h"

namespace py = pybind11;

template<typename T>
void register_sumtree(py::module& m, const char* name) {
  py::class_<SumTree<T>>(m, name)
    .def(py::init<const size_t&>())
    .def("update_value", &SumTree<T>::updateValue)
    .def("update_values", &SumTree<T>::updateValues)
    .def("get_index", &SumTree<T>::getIndex)
    .def("get_indices", &SumTree<T>::getIndices)
    .def("get_value", &SumTree<T>::getValue)
    .def("get_values", &SumTree<T>::getValues)
    .def("get_capacity", &SumTree<T>::getCapacity)
    .def("get_total_val", &SumTree<T>::getTotalVal)
    .def("__repr__",
         [](const SumTree<T>& st) {
           return "<SumTree(capacity=" + std::to_string(st.getCapacity()) + ", maxval=" + std::to_string(st.getTotalVal()) + ")>";
         });
}

PYBIND11_MODULE(SumTree, m) {
  register_sumtree<float>(m, "SumTreef");
}
