%module(directors="1") "pyextfun"

%{
#include "extfun.hh"
%}

#if defined(SWIGJAVA) || defined(SWIGPYTHON)
#define DBXML_DIRECTOR_CLASSES
#endif

#if defined(DBXML_DIRECTOR_CLASSES)

%feature("director") XmlExternalFunction;
class XmlExternalFunction
{
protected:
	XmlExternalFunction() {}
public:
	virtual ~XmlExternalFunction() {}
	virtual XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const = 0;
	virtual void close() = 0;
};

#endif



class MyExternalFunctionPow : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close();

};

class MyExternalFunctionSqrt : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close();
};



// from dbxml_python.i
// TBD -- see if there's a better way to extract
// info from exception
%{
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

// create the appropriate exception object
static void makeXmlException(const XmlException &xe)
{
	const char *ename = NULL;
	switch (xe.getExceptionCode()) {
	case XmlException::INTERNAL_ERROR:
		ename = "XmlInternalError"; break;
	case XmlException::CONTAINER_OPEN:
		ename = "XmlContainerOpen"; break;
	case XmlException::CONTAINER_CLOSED:
		ename = "XmlContainerClosed"; break;
	case XmlException::CONTAINER_EXISTS:
		ename = "XmlContainerExists"; break;
	case XmlException::CONTAINER_NOT_FOUND:
		ename = "XmlContainerNotFound"; break;
	case XmlException::NULL_POINTER:
		ename = "XmlNullPointer"; break;
	case XmlException::INDEXER_PARSER_ERROR:
		ename = "XmlParserError"; break;
	case XmlException::DATABASE_ERROR:
		ename = "XmlDatabaseError"; break;
	case XmlException::QUERY_PARSER_ERROR:
		ename = "XmlQueryParserError"; break;
	case XmlException::QUERY_EVALUATION_ERROR:
		ename = "XmlQueryEvaluationError"; break;
	case XmlException::LAZY_EVALUATION:
		ename = "XmlLazyEvaluation"; break;
	case XmlException::UNKNOWN_INDEX:
		ename = "XmlUnknownIndex"; break;
	case XmlException::DOCUMENT_NOT_FOUND:
		ename = "XmlDocumentNotFound"; break;
	case XmlException::INVALID_VALUE:
		ename = "XmlInvalidValue"; break;
	case XmlException::VERSION_MISMATCH:
		ename = "XmlVersionMismatch"; break;
	case XmlException::TRANSACTION_ERROR:
		ename = "XmlTransactionError"; break;
	case XmlException::UNIQUE_ERROR:
		ename = "XmlUniqueError"; break;
	case XmlException::NO_MEMORY_ERROR:
		ename = "XmlNoMemoryError"; break;
	case XmlException::EVENT_ERROR:
		ename = "XmlEventError"; break;
	case XmlException::OPERATION_INTERRUPTED:
		ename = "XmlOperationInterrupted"; break;
	case XmlException::OPERATION_TIMEOUT:
		ename = "XmlOperationTimeout"; break;
	default:
		ename = "XmlException";
	}
	if (ename != NULL) {
		PyObject *dbxmlMod = PyImport_ImportModule("dbxml");
		
		// set the value to an object with the code and text
		XmlException::ExceptionCode ec = xe.getExceptionCode();
		const char *what = xe.what();
		int dberr = xe.getDbErrno();
		int qline = xe.getQueryLine();
		int qcol = xe.getQueryColumn();

		// construct an exception object
		PyObject *errClass = PyObject_GetAttrString(dbxmlMod, ename);
		if (!errClass) {
			std::string msg = "Couldn't get BDB XML exception class: ";
			msg += ename;
			PyErr_SetString(PyExc_RuntimeError, msg.c_str());
			return;
		}
		PyObject *args = NULL;
		if (ec == XmlException::DATABASE_ERROR)
			args = Py_BuildValue( "(si)", what,dberr);
		else if ((ec == XmlException::QUERY_PARSER_ERROR) ||
			 (ec == XmlException::QUERY_EVALUATION_ERROR))
			args = Py_BuildValue( "(sii)", what,qline,qcol);
		else
			args = Py_BuildValue( "(s)", what);
		PyObject *errReturn = PyObject_CallObject(errClass, args);
		if (!errReturn) {
			std::string msg = "Couldn't instantiate BDB XML exception: ";
			msg += ename;
			PyErr_SetString(PyExc_RuntimeError, msg.c_str());
			return;
		}
		
		// set the actual error/exception object
		PyObject *etype = PyObject_Type(errReturn);
		PyErr_SetObject(etype, errReturn);
		Py_DECREF(args);
		Py_DECREF(errReturn);
		Py_DECREF(etype);
		Py_DECREF(errClass);
		Py_DECREF(dbxmlMod);
	}
}
%}

// if a director call fails (in python), throw
// an XmlException -- most of these calls originate
// in BDB XML proper, not Python.
%feature("director:except") {
	if ($error != NULL) {
		throwPyUserException();
	}
}


// Encapsulate release/acquire of global interpreter lock in
// an exception-safe class
// GMF: Added this for possible use, but it appears that the
// SWIG macros used with the -threads directive work well enough.
// Leave it for now, just in case it comes in handy.
%{
#ifdef SWIG_PYTHON_USE_GIL
class dbxml_releaseGIL {
public:
	dbxml_releaseGIL() {
		_save = PyEval_SaveThread();
	}
	~dbxml_releaseGIL() {
		PyEval_RestoreThread(_save);
	}
	PyThreadState *_save;
};
#else
class dbxml_releaseGIL {
public:
	dbxml_releaseGIL() {}
};
#endif
%}

%exception {
	try {
		$action
	} catch (XmlException &e) {
		SWIG_PYTHON_THREAD_END_ALLOW;
		makeXmlException(e);
		return NULL;
	}
}


