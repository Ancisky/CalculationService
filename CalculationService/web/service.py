import json
import numpy as np
from CalculationService.web.result import success,fail,rebut,error,result
from compute import c_net


class Net(object):
    def compute(request):
        json_data = request.body.decode('utf-8')
        if not json_data:
            return rebut("数据格式错误，请访问/cal/help寻求帮助！")
        data = json.loads(json_data)
        try:
            prop = data["prop"]
            x = np.array([[data['Mg'], data['Al'], data['Fe'],
            data['Co'], data['Ni'], data['Cu'],
            data['Zn'], data['Ga'], data['Ag'],
            data['In'], data['Sn'], data['Bi']]])
        except Exception as e:
            return rebut("数据格式错误，请访问/cal/help寻求帮助！")
        prediction = c_net.predict(prop, x)
        if prediction < 0:
            return fail("暂不支持此属性计算", "-")
        else:
            return success(data = prediction[0])