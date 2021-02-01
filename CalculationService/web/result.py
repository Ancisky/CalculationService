from django.http import JsonResponse, HttpResponse


class Status(object):
    # 操作成功
    CODE_SUCCESS=1
    MSG_SUCCESS='Success'
    # 操作失败
    CODE_FAIL=0
    MSG_FAIL='Fail'
    # 请求驳回
    CODE_REBUT=-1
    MSG_REBUT='Rebut'
    # 系统异常
    CODE_SYSTEM_ERROR=-1000
    MSG_SYSTEM_ERROR='System Error'

def result(code, message, data, kwargs=None):
    json_dict = {"code":code, "msg":message,"data":data}
    if kwargs and isinstance(kwargs, dict) and kwargs.keys():
        json_dict.update(kwargs)
    return JsonResponse(json_dict)

def success(message=Status.MSG_SUCCESS,data=None):
    return result(code=Status.CODE_SUCCESS, message=message, data=data)
def fail(message=Status.MSG_FAIL,data=None):
    return result(code=Status.CODE_FAIL, message=message, data=data)
def rebut(message=Status.MSG_REBUT,data=None):
    return result(code=Status.CODE_REBUT, message=message, data=data)
def error(message=Status.MSG_SYSTEM_ERROR,data=None):
    return result(code=Status.CODE_SYSTEM_ERROR, message=message, data=data)