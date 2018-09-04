import os.path as opath
import sys, platform
from shutil import copyfile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


def gen_cFile(prefix):
    py_fn, pyx_fn, c_fn = [prefix + extention for extention in ['.py', '.pyx', '.c']]
    if not opath.exists(pyx_fn):
        copyfile(py_fn, pyx_fn)
    else:
        if opath.getctime(pyx_fn) < opath.getmtime(py_fn):
            copyfile(py_fn, pyx_fn)
    if opath.exists(c_fn):
        if opath.getctime(c_fn) < opath.getmtime(pyx_fn):
            cythonize(prefix)
    else:
        cythonize(prefix)


def cythonize(fileName):
    ext_modules = [
        Extension('%s' % fileName,
                  ['%s.pyx' % fileName], include_dirs=['.']),
    ]
    setup(
        name='crowdsourcing_logistics',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules,
        script_args=['build_ext'],
        options={'build_ext': {'inplace': True, 'force': True}}
    )
    print('******** CYTHON COMPLETE ******')


def terminal_exec():
    plf = platform.platform()
    if plf.startswith('Linux'):
        # Linux server
        args = sys.argv
        if len(args) == len(['pyFile', 'fileName']):
            _, fileName = args
            cythonize(fileName)
        else:
            print('******** Error ******')
            print('******** Type packageName and fileName ******')
    elif plf.startswith('Darwin'):
        # Mac
        fileName = 'gh_mBundling'
        cythonize(fileName)
    else:
        # Window ?
        pass



if __name__ == '__main__':
    cythonize('gh_mBundling')