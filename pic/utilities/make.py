# coding: utf-8

import os
from os import listdir
from os.path import isfile, join
import importlib

from pyccel import epyccel

######################################################
import ast
import inspect

def get_types_decorator(cls):
    target = cls
    decorators = {}

    def visit_FunctionDef(node):
        for n in node.decorator_list:
            name = ''
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id

            if name == 'types':
                decorators[node.name] = [i.s for i in n.args]

    node_iter = ast.NodeVisitor()
    node_iter.visit_FunctionDef = visit_FunctionDef
    node_iter.visit(ast.parse(inspect.getsource(target)))
    return decorators
######################################################

files = [f for f in listdir('.') if isfile(join('.', f)) and
         f.split('.')[-1] == 'py' and  not( os.path.basename(f) == 'make.py' )]

library_files = ['bsplines.py', 'algebra.py']

libname = 'coco'

def compile_py(fname, compiler='gfortran', flags='-fPIC -O2 -c'):
    # ... run pyccel
    cmd = 'pyccel -t {fname}'.format(fname=fname)
    print(cmd)
    os.system(cmd)
    # ...

    # ... compile fortran code
    fname = os.path.basename(fname).split('.')[0] + '.f90'
    cmd = '{compiler} {flags} {fname}'
    cmd = cmd.format(compiler=compiler, flags=flags, fname=fname)
    print(cmd)
    os.system(cmd)
    # ...

def make_library(compiler='gfortran'):
    mypath = '.'
    files = library_files

    for f in files:
        compile_py(f, compiler=compiler)
#        make_header(f)

    fnames = [os.path.basename(f).split('.')[0] for f in files]
    files  = ['{}.o'.format(f) for f in fnames]
    files  = ' '.join(f for f in files)

    cmd = 'ar -r lib{libname}.a {files}'.format(files=files, libname=libname)
    os.system(cmd)

def make_clean():
    cmd = 'rm -f *.f90 *.o *.mod *.a *.pyh *.so'
    os.system(cmd)

def make_header(fname):
    fname = os.path.basename(fname).split('.')[0]
    mod = importlib.import_module(fname)
    decs = get_types_decorator(mod)

    lines = []
    pattern = '# header {func}({args})'
    for name,types in decs.items():
        args = ', '.join(i for i in types)
        line = pattern.format(func=name, args=args)
        lines += [line]

    fname = '{fname}.pyh'.format(fname=fname)
    f = open(fname, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()

def make_f2py(compiler='gnu95'):
    import core as mod
    mod = epyccel(mod, libs=[libname], libdirs=[os.getcwd()], openmp=True,
                  compiler=compiler)


#make_clean()
#make_library(compiler='ifort')
make_f2py(compiler='ifort')
