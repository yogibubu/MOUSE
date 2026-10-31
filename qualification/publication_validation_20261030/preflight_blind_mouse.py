"""Import/API/assignment preflight only: stop before the first electronic call."""
from pathlib import Path
import socket,ast,json
assert socket.getfqdn()=='oracle.sns.it'
import blind_mouse as b
class BeforeElectronicCall(Exception):pass
class StopBeforePrepare(ast.NodeTransformer):
 def visit_Expr(self,node):
  call=node.value
  if isinstance(call,ast.Call) and isinstance(call.func,ast.Subscript) and isinstance(call.func.value,ast.Name) and call.func.value.id=='env' and isinstance(call.func.slice,ast.Constant) and call.func.slice.value=='prepare':
   return ast.copy_location(ast.Raise(exc=ast.Call(func=ast.Name(id='BeforeElectronicCall',ctx=ast.Load()),args=[],keywords=[]),cause=None),node)
  return self.generic_visit(node)
tree=ast.parse(Path(b.__file__).read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker');module=StopBeforePrepare().visit(ast.Module(body=[worker],type_ignores=[]));ast.fix_missing_locations(module)
b.__dict__['BeforeElectronicCall']=BeforeElectronicCall;exec(compile(module,b.__file__,'exec'),b.__dict__)
try:b.worker('Diazirine')
except BeforeElectronicCall:
 result={'host':socket.getfqdn(),'status':'passed','scope':'imports, public root assignment, normalization and prepare API extraction; stopped before every electronic call','script_sha256':b.digest(Path(b.__file__))}
 b.write(Path(b.__file__).parent/'blind/mouse/preflight.json',result);print(json.dumps(result,indent=2))
else:raise RuntimeError('Preflight did not reach guarded entry point')
