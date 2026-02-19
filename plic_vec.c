#include <Python.h>
#include <math.h>

static PyObject* vec_norm(PyObject* self, PyObject* args) {
    PyObject* list;
    if (!PyArg_ParseTuple(args, "O", &list)) return NULL;
    if (!PyList_Check(list)) return NULL;

    Py_ssize_t n = PyList_Size(list);
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(list, i);
        double val = PyFloat_AsDouble(item);
        sum += val * val;
    }
    return PyFloat_FromDouble(sqrt(sum));
}

static PyObject* vec_dot(PyObject* self, PyObject* args) {
    PyObject* a;
    PyObject* b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    if (!PyList_Check(a) || !PyList_Check(b)) return NULL;

    Py_ssize_t n = PyList_Size(a);
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        double va = PyFloat_AsDouble(PyList_GetItem(a, i));
        double vb = PyFloat_AsDouble(PyList_GetItem(b, i));
        sum += va * vb;
    }
    return PyFloat_FromDouble(sum);
}

static PyMethodDef methods[] = {
    {"vec_norm", vec_norm, METH_VARARGS, "Compute vector norm"},
    {"vec_dot", vec_dot, METH_VARARGS, "Compute vector dot product"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "plic_vec",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_plic_vec(void) {
    return PyModule_Create(&module);
}