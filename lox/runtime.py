import builtins
from dataclasses import dataclass
from types import BuiltinFunctionType, FunctionType
from typing import TYPE_CHECKING

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

    def __call__(self, *args):
        # "Chamar" uma classe em Lox cria uma nova instância dela.
        # Por enquanto, ignoramos os argumentos, pois ainda não temos inicializadores (init).
        instance = LoxInstance(klass=self)
        return instance

    def __str__(self) -> str:
        return self.name

@dataclass
class LoxInstance:
    """Representa uma instância de uma classe Lox."""
    klass: LoxClass

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

    def call(self, args: list["Value"]):
        if len(args) != len(self.params):
            raise TypeError(f"'{self.name}' esperava {len(self.params)} argumentos, mas recebeu {len(args)}.")
        
        local_env = dict(zip(self.params, args))
        call_ctx = self.ctx.push(local_env)

        try:
            self.body.eval(call_ctx)
        except LoxReturn as ex:
            return ex.value
        
        return None

    def __call__(self, *args):
        return self.call(list(args))

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