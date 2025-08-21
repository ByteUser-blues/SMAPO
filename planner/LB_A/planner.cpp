// cppimport
#include "planner.h"

PYBIND11_MODULE(planner, m) {
    py::class_<planner>(m, "planner")
            .def(py::init<std::vector<std::vector<int>>, bool, bool, bool>())
            .def("set_abs_start", &planner::set_abs_start)
            .def("update_path", &planner::update_path)
            .def("get_path", &planner::get_path)
            .def("get_next_node", &planner::get_next_node)
            .def("precompute_penalty_matrix", &planner::precompute_penalty_matrix)
            .def("set_penalties", &planner::set_penalties)
            .def("update_occupations", &planner::update_occupations)
            .def("get_bfs_map", &planner::get_bfs_map)
            .def("update_cur_map", &planner::update_cur_map);
    
    py::class_<Grid_T_MAP>(m, "Grid_T_MAP")
        .def(py::init<int, int, int>())
        .def_readwrite("i", &Grid_T_MAP::i)
        .def_readwrite("j", &Grid_T_MAP::j)
        .def_readwrite("t", &Grid_T_MAP::t)
        .def("__eq__", &Grid_T_MAP::operator==)
        .def("__hash__", [](const Grid_T_MAP& p) {
            return Grid_T_MAP::Hash{}(p);
        });
    
    py::class_<std::unordered_map<Grid_T_MAP, int, Grid_T_MAP::Hash>>(m, "Grid_T_hash")
        .def(py::init<>())
        .def("__getitem__", [](std::unordered_map<Grid_T_MAP, int, Grid_T_MAP::Hash>& m, const Grid_T_MAP& p) {
            return m[p];
        })
        .def("__setitem__", [](std::unordered_map<Grid_T_MAP, int, Grid_T_MAP::Hash>& m, const Grid_T_MAP& p, int value) {
            m[p] = value;
        })
        .def("size", &std::unordered_map<Grid_T_MAP, int, Grid_T_MAP::Hash>::size)
        .def("clear", &std::unordered_map<Grid_T_MAP, int, Grid_T_MAP::Hash>::clear);
}

<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>