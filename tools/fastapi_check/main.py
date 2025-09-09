import numpy as np
from typing import Annotated
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, Header, Depends, HTTPException

app = FastAPI()


def read_arr_from_rawfile(path="$HOME/Downloads/arr", dtype_="uint8", shape_="512,512,3" ):
    """
    this suceeded to recover the numpy array from octet-stream of bytes download from /arr endpoint, 
    but it is cheating regards the metadata 
    """
    x = None

    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,shape_.split(","))) 

    with open(os.path.expandvars(path), "rb") as fp:
        xb = fp.read()
        x = np.frombuffer(xb, dtype=dtype ).reshape(*shape)
    pass
    return x



def array_create_():
    a = np.zeros((512, 512, 3), dtype=np.uint8)
    a[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    return a



def make_array_response( a: np.array, level : int = 0, media_type : str = "application/octet-stream" ):
    headers = {} 
    headers["x-numpy-level"] = str(level)
    headers["x-numpy-dtype"] = a.dtype.name
    headers["x-numpy-shape"] = str(a.shape)
    return Response(a.tobytes('C'), headers=headers, media_type=media_type )



async def parse_request_as_array(request: Request):
    """
    :param request: FastAPI Request
    :return arr: NumPy array 

    Uses request body and headers with array dtype and shape to reconstruct the uploaded NumPy array

    DONE: check token in headers and return not auth when its missing/wrong 

    TODO: get level from query parameter, not header https://fastapi.tiangolo.com/tutorial/query-params/#optional-parameters

    https://fastapi.tiangolo.com/tutorial/header-params/#automatic-conversion

    https://fastapi.tiangolo.com/tutorial/header-param-models/



    https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status

    Although the HTTP standard specifies "unauthorized", semantically this response 401
    means "unauthenticated". That is, the client must authenticate itself to get
    the requested response.

    """
    token_ = request.headers.get('x-numpy-token')
    if token_ != "secret":
        raise HTTPException(status_code=401, detail="x-numpy-token invalid")
    pass

    level_ = request.headers.get('x-numpy-level','0')
    dtype_ = request.headers.get('x-numpy-dtype')
    shape_ = request.headers.get('x-numpy-shape')

    level = int(level_)
    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,filter(None,map(str.strip,shape_.replace("(","").replace(")","").split(",")))))

    data: bytes = await request.body()
    a = np.frombuffer(data, dtype=dtype ).reshape(*shape)

    if level > 0:
        print("[parse_request_as_array")
        #print(" x_numpy_token[%s]" % x_numpy_token )
        print(" token_[%s]" % token_ )
        print(" level[%d]" % level )
        print(request.headers) 
        #print("data[%s]" % data )
        print("dtype_[%s]" % str(dtype) )
        print("shape_[%s]" % str(shape_) )
        print("shape[%s]"  % str(shape) )
        print("a[%s]" % a )
        print("]parse_request_as_array")
    pass
    return a




#async def verify_token(x_numpy_token: Annotated[str | None, Header()]):
#    print("verify_token x_numpy_token[%s]" % x_numpy_token) 
#    if x_numpy_token != "secret":
#        raise HTTPException(status_code=400, detail="x-numpy-token invalid")
#
#@app.post('/array_transform', response_class=Response, dependencies=[Depends(verify_token),])

@app.post('/array_transform', response_class=Response)
async def array_transform(a: np.array = Depends(parse_request_as_array)):
    """
    :param a:
    :return response: Response

    1. parse_request_as_array providing the uploaded NumPy *a*
    2. operate on *a* giving *b*
    3. return *b* as FastAPI Response
    """

    #b = a * 10.   # do some operation on the input array  

    #b = np.repeat(a * 10,2)

    b = a + 1

    return make_array_response(b)


@app.get('/array_create', response_class=Response)
def array_create():
    """
    zeta:Downloads blyth$ curl -s -D /dev/stdout http://127.0.0.1:8000/array_create  --output arr
    HTTP/1.1 200 OK
    date: Tue, 09 Sep 2025 03:08:11 GMT
    server: uvicorn
    x-numpy-level: 0
    x-numpy-dtype: uint8
    x-numpy-shape: 512,512,3
    content-length: 786432
    content-type: application/octet-stream

    zeta:Downloads blyth$ l arr
    1536 -rw-r--r--  1 blyth  staff  786432 Sep  9 11:08 arr

    zeta:Downloads blyth$ echo $(( 512*512*3 ))
    786432
    """
    a = array_create_()
    return make_array_response( a )


