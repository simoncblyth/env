// included into extfun_python.i
static void throwPyUserException() {
	PyObject *type = 0;
	PyObject *value = 0;
	PyObject *traceback = 0;
	PyErr_Fetch(&type, &value, &traceback);
	if (value) {
		char buf[1024];
		PyObject *str = PyObject_Str(value);
		Py_XINCREF(type);
		PyErr_Clear();
		PyOS_snprintf(buf, sizeof(buf), "Error from Python user code: %s", PyString_AsString(str));
		Py_DECREF(str);
		//PyErr_Print();
		throw XmlException(XmlException::INTERNAL_ERROR,
				   buf);
	}
}


