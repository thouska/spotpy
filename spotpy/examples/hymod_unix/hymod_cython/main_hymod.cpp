#include <iostream>
#include "Python.h"
#include "hymod.h"

int main(int argc, char *argv[])
{
    
    //PyInit_hymod();  //Needed! called PyInit_hello() for Python3

    int status = PyImport_AppendInittab("hymod", PyInit_hymod);

    
    
    if (status == -1)
    {
        std::cout << "We got a error " << status << std::endl;
        return -1; //error
    }

    Py_Initialize(); //Needed!

    PyObject *module = PyImport_ImportModule("hymod");
    //std::cout << "Module is: " << module << std::endl;

    if (module == NULL)
    {
        Py_Finalize();
        return -1; //error
    }
    
    hymod_run();


    Py_Finalize(); //Needed!

    return 0;
}