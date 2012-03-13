
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
