#include "ufunc.h"

#include "xsf_special.h"

using namespace std;

using func_f_f1f1_t =
    void (*)(float, mdspan<float, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<float, dextents<ptrdiff_t, 1>, layout_stride>);
using func_d_d1d1_t =
    void (*)(double, mdspan<double, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<double, dextents<ptrdiff_t, 1>, layout_stride>);
using func_F_F1F1_t =
    void (*)(complex<float>, mdspan<complex<float>, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<complex<float>, dextents<ptrdiff_t, 1>, layout_stride>);
using func_D_D1D1_t =
    void (*)(complex<double>, mdspan<complex<double>, dextents<ptrdiff_t, 1>, layout_stride>, mdspan<complex<double>, dextents<ptrdiff_t, 1>, layout_stride>);

using func_f_f2f2_t =
    void (*)(float, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>);
using func_d_d2d2_t =
    void (*)(double, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>);
using func_F_F2F2_t =
    void (*)(complex<float>, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>);
using func_D_D2D2_t =
    void (*)(complex<double>, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>);

using func_fb_f2f2_t =
    void (*)(float, bool, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<float, dextents<ptrdiff_t, 2>, layout_stride>);
using func_db_d2d2_t =
    void (*)(double, bool, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<double, dextents<ptrdiff_t, 2>, layout_stride>);

using func_Flb_F2F2_t =
    void (*)(complex<float>, long, bool, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>);
using func_Dlb_D2D2_t =
    void (*)(complex<double>, long, bool, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>);

using func_ff_F2_t = void (*)(float, float, mdspan<complex<float>, dextents<ptrdiff_t, 2>, layout_stride>);
using func_dd_D2_t = void (*)(double, double, mdspan<complex<double>, dextents<ptrdiff_t, 2>, layout_stride>);

extern const char *lpn_doc;
extern const char *lpmn_doc;
extern const char *clpmn_doc;
extern const char *lqn_doc;
extern const char *lqmn_doc;
extern const char *rctj_doc;
extern const char *rcty_doc;
extern const char *sph_harm_all_doc;

// This is needed by sf_error, it is defined in the Cython "_ufuncs_extra_code_common.pxi" for "_generate_pyx.py".
// It exists to "call PyUFunc_getfperr in a context where PyUFunc_API array is initialized", but here we are
// already in such a context.
extern "C" int wrap_PyUFunc_getfperr() { return PyUFunc_getfperr(); }

static PyModuleDef _gufuncs_def = {
    PyModuleDef_HEAD_INIT,
    "_gufuncs",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__gufuncs() {
    import_array();
    import_umath();
    if (PyErr_Occurred()) {
        return NULL;
    }

    PyObject *_gufuncs = PyModule_Create(&_gufuncs_def);
    if (_gufuncs == nullptr) {
        return NULL;
    }

#if Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(_gufuncs, Py_MOD_GIL_NOT_USED);
#endif


    PyObject *legendre_p_all = Py_BuildValue(
        "(N, N, N)",
        SpecFun_NewGUFunc({static_cast<func_d_d1_t>(::legendre_p_all), static_cast<func_f_f1_t>(::legendre_p_all),
                           static_cast<func_D_D1_t>(::legendre_p_all), static_cast<func_F_F1_t>(::legendre_p_all)},
                          1, "legendre_p_all", nullptr, "()->(np1)", legendre_map_dims<1>),
        SpecFun_NewGUFunc({static_cast<func_d_d1d1_t>(::legendre_p_all), static_cast<func_f_f1f1_t>(::legendre_p_all),
                           static_cast<func_D_D1D1_t>(::legendre_p_all), static_cast<func_F_F1F1_t>(::legendre_p_all)},
                          2, "legendre_p_all", nullptr, "()->(np1),(np1)", legendre_map_dims<2>),
        SpecFun_NewGUFunc(
            {static_cast<func_d_d1d1d1_t>(::legendre_p_all), static_cast<func_f_f1f1f1_t>(::legendre_p_all),
             static_cast<func_D_D1D1D1_t>(::legendre_p_all), static_cast<func_F_F1F1F1_t>(::legendre_p_all)},
            3, "legendre_p_all", nullptr, "()->(np1),(np1),(np1)", legendre_map_dims<3>));
    PyModule_AddObjectRef(_gufuncs, "legendre_p_all", legendre_p_all);

    // key is norm, diff_n
    PyObject *assoc_legendre_p_all = Py_BuildValue(
        "{(O, i): N, (O, i): N, (O, i): N, (O, i): N, (O, i): N, (O, i): N}", Py_True, 0,
        SpecFun_NewGUFunc({[](double z, long long int branch_cut, double_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res);
                           },
                           [](float z, long long int branch_cut, float_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res);
                           },
                           [](cdouble z, long long int branch_cut, cdouble_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res);
                           },
                           [](cfloat z, long long int branch_cut, cfloat_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res);
                           }},
                          1, "assoc_legendre_p_all", nullptr, "(),()->(np1,mpmp1)", assoc_legendre_map_dims<1>),
        Py_True, 1,
        SpecFun_NewGUFunc({[](double z, long long int branch_cut, double_2d res, double_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac);
                           },
                           [](float z, long long int branch_cut, float_2d res, float_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac);
                           },
                           [](cdouble z, long long int branch_cut, cdouble_2d res, cdouble_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac);
                           },
                           [](cfloat z, long long int branch_cut, cfloat_2d res, cfloat_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac);
                           }},
                          2, "assoc_legendre_p_all", nullptr, "(),()->(np1,mpmp1),(np1,mpmp1)",
                          assoc_legendre_map_dims<2>),
        Py_True, 2,
        SpecFun_NewGUFunc(
            {[](double z, long long int branch_cut, double_2d res, double_2d res_jac, double_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac, res_hess);
             },
             [](float z, long long int branch_cut, float_2d res, float_2d res_jac, float_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac, res_hess);
             },
             [](cdouble z, long long int branch_cut, cdouble_2d res, cdouble_2d res_jac, cdouble_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac, res_hess);
             },
             [](cfloat z, long long int branch_cut, cfloat_2d res, cfloat_2d res_jac, cfloat_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_norm, z, branch_cut, res, res_jac, res_hess);
             }},
            3, "assoc_legendre_p_all", nullptr, "(),()->(np1,mpmp1),(np1,mpmp1),(np1,mpmp1)",
            assoc_legendre_map_dims<3>),
        Py_False, 0,
        SpecFun_NewGUFunc({[](double z, long long int branch_cut, double_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res);
                           },
                           [](float z, long long int branch_cut, float_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res);
                           },
                           [](cdouble z, long long int branch_cut, cdouble_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res);
                           },
                           [](cfloat z, long long int branch_cut, cfloat_2d res) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res);
                           }},
                          1, "assoc_legendre_p_all", nullptr, "(),()->(np1,mpmp1)", assoc_legendre_map_dims<1>),
        Py_False, 1,
        SpecFun_NewGUFunc({[](double z, long long int branch_cut, double_2d res, double_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac);
                           },
                           [](float z, long long int branch_cut, float_2d res, float_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac);
                           },
                           [](cdouble z, long long int branch_cut, cdouble_2d res, cdouble_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac);
                           },
                           [](cfloat z, long long int branch_cut, cfloat_2d res, cfloat_2d res_jac) {
                               ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac);
                           }},
                          2, "assoc_legendre_p_all", nullptr, "(),()->(np1,mpmp1),(np1,mpmp1)",
                          assoc_legendre_map_dims<2>),
        Py_False, 2,
        SpecFun_NewGUFunc(
            {[](double z, long long int branch_cut, double_2d res, double_2d res_jac, double_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac, res_hess);
             },
             [](float z, long long int branch_cut, float_2d res, float_2d res_jac, float_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac, res_hess);
             },
             [](cdouble z, long long int branch_cut, cdouble_2d res, cdouble_2d res_jac, cdouble_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac, res_hess);
             },
             [](cfloat z, long long int branch_cut, cfloat_2d res, cfloat_2d res_jac, cfloat_2d res_hess) {
                 ::assoc_legendre_p_all(assoc_legendre_unnorm, z, branch_cut, res, res_jac, res_hess);
             }},
            3, "assoc_legendre_p_all", nullptr, "(),()->(np1,mpmp1),(np1,mpmp1),(np1,mpmp1)",
            assoc_legendre_map_dims<3>));
    PyModule_AddObjectRef(_gufuncs, "assoc_legendre_p_all", assoc_legendre_p_all);

    PyObject *_clpmn = SpecFun_NewGUFunc(
        {static_cast<func_Flb_F2F2_t>(special::clpmn), static_cast<func_Dlb_D2D2_t>(special::clpmn)}, 2, "_clpmn",
        clpmn_doc, "(),(),()->(mp1,np1),(mp1,np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_clpmn", _clpmn);

    PyObject *_lqn = SpecFun_NewGUFunc({static_cast<func_d_d1d1_t>(xsf::lqn), static_cast<func_f_f1f1_t>(xsf::lqn),
                                        static_cast<func_D_D1D1_t>(xsf::lqn), static_cast<func_F_F1F1_t>(xsf::lqn)},
                                       2, "_lqn", lqn_doc, "()->(np1),(np1)", legendre_map_dims<2>);
    PyModule_AddObjectRef(_gufuncs, "_lqn", _lqn);

    PyObject *_lqmn = SpecFun_NewGUFunc({static_cast<func_d_d2d2_t>(xsf::lqmn), static_cast<func_f_f2f2_t>(xsf::lqmn),
                                         static_cast<func_D_D2D2_t>(xsf::lqmn), static_cast<func_F_F2F2_t>(xsf::lqmn)},
                                        2, "_lqmn", lqmn_doc, "()->(mp1,np1),(mp1,np1)", assoc_legendre_map_dims<2>);
    PyModule_AddObjectRef(_gufuncs, "_lqmn", _lqmn);

    PyObject *sph_harm_y_all = Py_BuildValue(
        "(N,N,N)",
        SpecFun_NewGUFunc({static_cast<func_dd_D2_t>(::sph_harm_y_all), static_cast<func_ff_F2_t>(::sph_harm_y_all)}, 1,
                          "sph_harm_y_all", nullptr, "(),()->(np1,mpmp1)", sph_harm_map_dims<1>),
        SpecFun_NewGUFunc(
            {static_cast<func_dd_D2D3_t>(::sph_harm_y_all), static_cast<func_ff_F2F3_t>(::sph_harm_y_all)}, 2,
            "sph_harm_y_all", nullptr, "(),()->(np1,mpmp1),(2,np1,mpmp1)", sph_harm_map_dims<2>),
        SpecFun_NewGUFunc(
            {static_cast<func_dd_D2D3D4_t>(::sph_harm_y_all), static_cast<func_ff_F2F3F4_t>(::sph_harm_y_all)}, 3,
            "sph_harm_y_all", nullptr, "(),()->(np1,mpmp1),(2,np1,mpmp1),(2,2,np1,mpmp1)", sph_harm_map_dims<3>));
    PyModule_AddObjectRef(_gufuncs, "sph_harm_y_all", sph_harm_y_all);

    PyObject *_rctj = SpecFun_NewGUFunc({static_cast<func_d_d1d1_t>(xsf::rctj), static_cast<func_f_f1f1_t>(xsf::rctj)},
                                        2, "_rctj", rctj_doc, "()->(np1),(np1)", legendre_map_dims<2>);
    PyModule_AddObjectRef(_gufuncs, "_rctj", _rctj);

    PyObject *_rcty = SpecFun_NewGUFunc({static_cast<func_d_d1d1_t>(xsf::rcty), static_cast<func_f_f1f1_t>(xsf::rcty)},
                                        2, "_rcty", rcty_doc, "()->(np1),(np1)", legendre_map_dims<2>);
    PyModule_AddObjectRef(_gufuncs, "_rcty", _rcty);

    PyObject *_sph_harm_all = SpecFun_NewGUFunc(
        {static_cast<func_dd_D2_t>(special::sph_harm_all), static_cast<func_ff_F2_t>(special::sph_harm_all)}, 1,
        "_sph_harm_all", sph_harm_all_doc, "(),()->(mp1,np1)"
    );
    PyModule_AddObjectRef(_gufuncs, "_sph_harm_all", _sph_harm_all);

    return _gufuncs;
}
