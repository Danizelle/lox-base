"""
Implementa o transformador da árvore sintática que converte entre as representações

    lark.Tree -> lox.ast.Node.

A resolução de vários exercícios requer a modificação ou implementação de vários
métodos desta classe.
"""

from typing import Callable
from lark import Transformer, v_args

from . import runtime as op
from .ast import *


def op_handler(op: Callable):
    """
    Fábrica de métodos que lidam com operações binárias na árvore sintática.

    Recebe a função que implementa a operação em tempo de execução.
    """

    def method(self, left, right):
        return BinOp(left, right, op)

    return method


@v_args(inline=True)
class LoxTransformer(Transformer):
    # Programa
    def program(self, *stmts):
        return Program(list(stmts))

    # Operações matemáticas básicas
    mul = op_handler(op.mul)
    div = op_handler(op.truediv)
    sub = op_handler(op.sub)
    add = op_handler(op.add)

    # Comparações
    gt = op_handler(op.gt)
    lt = op_handler(op.lt)
    ge = op_handler(op.ge)
    le = op_handler(op.le)
    eq = op_handler(op.eq)
    ne = op_handler(op.ne)

    # Outras expressões
    def call(self, callee: Expr, params=None):
        return Call(callee, params or [])
        
    def params(self, *args):
        return list(args)

    def getattr(self, obj, name):
        return Getattr(obj, name.name)

    # Operadores unários
    def neg(self, operand):
        return UnaryOp(operand, op.neg) 

    def not_(self, operand):
        return UnaryOp(operand, op.not_)
    
    # Operadores lógicos
    def and_(self, left, right):
        return And(left, right)
    
    def or_(self, left, right):
        return Or(left, right)
    
    # Atribuições
    def assign(self, name: Var, value: Expr):
        return Assign(name.name, value)

    def setattr_assign(self, target, value):
        return Setattr(target.obj, target.name, value)
    
    # Comandos e literais
    def print_cmd(self, expr):
        return Print(expr)

    def var_decl(self, name, initializer=None):
        if initializer is None:
            initializer = Literal(None)
        return VarDef(name.name, initializer)

    def block(self, *stmts):
        return Block(list(stmts))
    
    def if_cmd(self, condition, then_branch, else_branch=None):
        if else_branch is None:
            else_branch = Block([]) # Bloco vazio se não houver else
        return If(condition, then_branch, else_branch)

    def while_cmd(self, condition, body):
        return While(condition, body)
    
    def expr_stmt(self, expr):
        return ExprStmt(expr)

    # Literais e Variáveis
    def VAR(self, token):
        name = str(token)
        return Var(name)

    def NUMBER(self, token):
        num = float(token)
        return Literal(num)
    
    def STRING(self, token):
        text = str(token)[1:-1]
        return Literal(text)
    
    def NIL(self, _):
        return Literal(None)

    def BOOL(self, token):
        return Literal(token == "true")

    # Tratamento de 'for' (desugaring para 'while')
    def empty_init(self):
        return None 

    def empty_cond(self):
        return Literal(True)

    def empty_incr(self):
        return None

    def for_cmd(self, init, cond, incr, body):
        # Constrói o corpo do while
        while_body = [body]
        if incr is not None:
            while_body.append(ExprStmt(incr))
        
        # Constrói o laço while
        if cond is None:
            cond = Literal(True)
        loop = While(cond, Block(while_body))

        # Adiciona o inicializador, se existir
        if init is not None:
            return Block([init, loop])
        
        return loop
    
    # Funções e retornos
    def return_stmt(self, value=None):
        return Return(value)

    def function_declaration(self, name, params, body):
        return Function(name.name, params or [], body)

    def fun_params(self, *params):
        return list(params)
    
    def this(self, _):
        return This()
    
    def THIS(self, token):
        return This()

    def super_getattr(self, super_token, name):
        return Super(name.name)
    
    def class_declaration(self, name, *args):
        """
        Transforma a declaração de uma classe.
        Aceita:
        - class A {}
        - class A { method1, method2, ... }
        - class A < B {}
        - class A < B { method1, method2, ... }
        """
        methods = []
        base = None
        
        # Se não há argumentos, é uma classe vazia
        if not args:
            return Class(name.name, [], None)
        
        # Verifica se há herança
        if len(args) >= 1 and isinstance(args[0], Var):
            # Primeiro argumento é a classe base
            base = args[0]
            # Métodos vêm depois
            methods = list(args[1:])
        else:
            # Todos os argumentos são métodos
            methods = list(args)
        
        return Class(name.name, methods, base)
    
    def method(self, name, params, body):
        """Transforma um método de classe."""
        return Function(name.name, params or [], body)