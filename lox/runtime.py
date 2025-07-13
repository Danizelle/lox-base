import builtins
from dataclasses import dataclass, field
from types import BuiltinFunctionType, FunctionType
from typing import TYPE_CHECKING, Any, Optional

from .ctx import Ctx

if TYPE_CHECKING:
    from .ast import Block, Value

class LoxError(Exception):
    """Exceção para erros de execução Lox."""

class LoxReturn(Exception):
    """Exceção usada para implementar o comando 'return' do Lox."""
    def __init__(self, value: "Value"):
        super().__init__()
        self.value = value

@dataclass
class LoxClass:
    """Representa uma classe Lox em tempo de execução."""
    name: str
    methods: dict[str, "LoxFunction"]
    base: Optional["LoxClass"] = None

    def __call__(self, *args):
        """
        self.__call__(x, y) <==> self(x, y)

        Em Lox, criamos instâncias de uma classe chamando-a como uma função.
        """
        instance = LoxInstance(self)
        
        # Se houver um método init, chame-o com os argumentos fornecidos
        try:
            initializer = self.get_method("init")
            bound_initializer = initializer.bind(instance)
            bound_initializer(*args)
        except LoxError:
            if len(args) > 0:
                raise TypeError(f"Esperava 0 argumentos mas recebeu {len(args)}")
            
        return instance

    def get_method(self, name: str) -> "LoxFunction":
        """
        Procura o método na classe atual ou em suas bases.
        Levanta LoxError se não encontrar.
        """
        # Procura na classe atual
        if name in self.methods:
            return self.methods[name]
        
        # Se não encontrou e tiver uma classe base, procura nela
        if self.base is not None:
            return self.base.get_method(name)
        
        # Se não encontrou em lugar nenhum, levanta exceção
        raise LoxError(f"Método '{name}' não encontrado na classe '{self.name}'")

    def __str__(self) -> str:
        return self.name

@dataclass
class LoxInstance:
    """Representa uma instância de uma classe Lox."""
    klass: LoxClass
    fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        """Implementa a busca de atributos em instâncias Lox."""
        if name in self.fields:
            return self.fields[name]
        
        try:
            method = self.klass.get_method(name)
            # Para o método init, usar comportamento especial
            if name == "init":
                bound_method = method.bind(self)
                return LoxInitFunction(
                    bound_method.name,
                    bound_method.params,
                    bound_method.body,
                    bound_method.ctx,
                    self
                )
            else:
                # Associa o método à instância atual
                return method.bind(self)
        except LoxError:
            raise LoxError(f"Campo '{name}' não existe")

    def get_field(self, name: str):
        """Retorna o valor de um campo da instância."""
        return self.__getattr__(name)

    def set_field(self, name: str, value):
        """Define o valor de um campo da instância."""
        self.fields[name] = value
        return value
    
    def __str__(self) -> str:
        return f"{self.klass.name} instance"

@dataclass
class LoxFunction:
    """Representa uma função Lox em tempo de execução."""
    name: str
    params: list[str]
    body: "Block"
    ctx: Ctx

    def __str__(self) -> str:
        if self.name:
            return f"<fn {self.name}>"
        return "<fn>"

    def bind(self, obj: "Value") -> "LoxFunction":
        """Associa essa função a um this específico."""
        # Cria uma nova função com o contexto que inclui o 'this'
        return LoxFunction(
            self.name,
            self.params,
            self.body,
            self.ctx.push({"this": obj})
        )

    def call(self, args: list["Value"]):
        # Converte os parâmetros para strings se forem objetos Var
        param_names = []
        for param in self.params:
            if hasattr(param, 'name'):
                param_names.append(param.name)
            else:
                param_names.append(str(param))
                
        if len(args) != len(param_names):
            raise TypeError(f"'{self.name}' esperava {len(param_names)} argumentos, mas recebeu {len(args)}.")

        local_env = dict(zip(param_names, args))

        # Se o método está sendo chamado como método de instância, o primeiro argumento é a instância (this)
        this = None
        supercls = None
        if hasattr(self, "bind_instance"):
            this = self.bind_instance
        elif self.params and self.params[0] == "this":
            # Heurística: se o primeiro parâmetro é 'this', provavelmente é um método
            this = args[0] if args else None
        # Tenta descobrir a superclasse
        if hasattr(self, "bind_superclass"):
            supercls = self.bind_superclass
        elif hasattr(self.ctx, "superclass"):
            supercls = self.ctx.superclass

        # Adiciona 'this' e 'super' ao contexto se disponíveis
        if this is not None or supercls is not None:
            special = {}
            if this is not None:
                special["this"] = this
            if supercls is not None:
                special["super"] = supercls
            call_ctx = self.ctx.push({**local_env, **special})
        else:
            call_ctx = self.ctx.push(local_env)

        try:
            self.body.eval(call_ctx)
        except LoxReturn as ex:
            return ex.value
        return None

    def __call__(self, *args):
        return self.call(list(args))

@dataclass
class LoxInitFunction(LoxFunction):
    """Representa um método init vinculado a uma instância."""
    instance: "LoxInstance"
    
    def __call__(self, *args):
        # Executa o método init normalmente
        super().call(list(args))
        # Mas sempre retorna a instância em vez do resultado
        return self.instance

# --- Funções de Semântica do Lox ---

def show(value: "Value") -> str:
    """Converte um valor Lox para sua representação em string."""
    if value is None:
        return "nil"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        # Dica: str(42.0) -> "42.0", removesuffix -> "42"
        return str(value).removesuffix('.0')
    if isinstance(value, (BuiltinFunctionType, FunctionType)):
        return "<native fn>"
    # Para LoxFunction, LoxClass, LoxInstance e str, o __str__ já faz o trabalho.
    return str(value)

def print(value: "Value"):
    """Imprime um valor Lox usando a representação correta."""
    builtins.print(show(value))

def truthy(value: "Value") -> bool:
    """
    Avalia o valor de acordo com as regras de veracidade do Lox.
    Apenas 'nil' e 'false' são considerados falsos.
    """
    if value is None or value is False:
        return False
    return True

def not_(value: "Value") -> bool:
    """Operador de negação Lox (!)."""
    return not truthy(value)

def neg(value: "Value") -> float:
    """Operador de negação aritmética Lox (-)."""
    if not isinstance(value, float):
        raise LoxError("Operand must be a number.")
    return -value

def eq(a: "Value", b: "Value") -> bool:
    """Operador de igualdade Lox (==)."""
    # Em Lox, tipos diferentes nunca são iguais.
    if type(a) is not type(b):
        return False
    return a == b

def ne(a: "Value", b: "Value") -> bool:
    """Operador de desigualdade Lox (!=)."""
    return not eq(a, b)

def _check_numbers(*operands):
    """Função auxiliar para garantir que todos os operandos são números."""
    for op in operands:
        if not isinstance(op, float):
            raise LoxError("Operands must be numbers.")

def add(a: "Value", b: "Value") -> "Value":
    """Operador de adição Lox (+)."""
    if isinstance(a, float) and isinstance(b, float):
        return a + b
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    raise LoxError("Operands must be two numbers or two strings.")

def sub(a: float, b: float) -> float:
    """Operador de subtração Lox (-)."""
    _check_numbers(a, b)
    return a - b

def mul(a: float, b: float) -> float:
    """Operador de multiplicação Lox (*)."""
    _check_numbers(a, b)
    return a * b

def truediv(a: float, b: float) -> float:
    """Operador de divisão Lox (/)."""
    _check_numbers(a, b)
    if b == 0:
        raise LoxError("Division by zero.")
    return a / b

def lt(a: float, b: float) -> bool:
    """Operador 'menor que' Lox (<)."""
    _check_numbers(a, b)
    return a < b

def le(a: float, b: float) -> bool:
    """Operador 'menor ou igual' Lox (<=)."""
    _check_numbers(a, b)
    return a <= b       

def gt(a: float, b: float) -> bool:
    """Operador 'maior que' Lox (>)."""
    _check_numbers(a, b)
    return a > b

def ge(a: float, b: float) -> bool:
    """Operador 'maior ou igual' Lox (>=)."""
    _check_numbers(a, b)
    return a >= b

# Lista de nomes a serem exportados para o transformer
__all__ = [
    "add", "sub", "mul", "truediv",
    "eq", "ne", "lt", "le", "gt", "ge",
    "neg", "not_",
    "truthy", "show", "print", "LoxError"
]