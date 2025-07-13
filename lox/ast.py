from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional

from .ctx import Ctx
from .runtime import LoxFunction, LoxReturn, LoxClass, LoxError, LoxInstance, print as lox_print

# Importações para a validação semântica
from .node import Node, Cursor
from .errors import SemanticError

# Palavras reservadas da linguagem Lox
RESERVED_WORDS = {
    "and", "class", "else", "false", "for", "fun", "if", "nil",
    "or", "print", "return", "super", "this", "true", "var", "while"
}

# TIPOS BÁSICOS

Value = bool | str | float | None


class Expr(Node, ABC):
    """Classe base para expressões."""

class Stmt(Node, ABC):
    """Classe base para comandos."""

@dataclass
class Program(Node):
    """Representa um programa."""
    stmts: list[Stmt]

    def eval(self, ctx: Ctx):
        for stmt in self.stmts:
            stmt.eval(ctx)

# EXPRESSÕES

@dataclass
class BinOp(Expr):
    """Uma operação infixa com dois operandos."""
    left: Expr
    right: Expr
    op: Callable[[Value, Value], Value]

    def eval(self, ctx: Ctx):
        left_value = self.left.eval(ctx)
        right_value = self.right.eval(ctx)
        return self.op(left_value, right_value)

@dataclass
class Var(Expr):
    """Uma variável no código."""
    name: str

    def eval(self, ctx: Ctx):
        try:
            return ctx[self.name]
        except KeyError:
            raise NameError(f"variável {self.name} não existe!")

@dataclass
class Literal(Expr):
    """Representa valores literais no código."""
    value: Value

    def eval(self, ctx: Ctx):
        return self.value

@dataclass
class ExprStmt(Stmt):
    expr: Expr
    def eval(self, ctx: Ctx):
        self.expr.eval(ctx)

@dataclass
class And(Expr):
    """Uma operação 'and'."""
    left: Expr
    right: Expr
    def eval(self, ctx: Ctx):
        left_val = self.left.eval(ctx)
        if left_val is False or left_val is None:
            return left_val
        return self.right.eval(ctx)

@dataclass
class Or(Expr):
    """Uma operação 'or'."""
    left: Expr
    right: Expr
    def eval(self, ctx: Ctx):
        left_val = self.left.eval(ctx)
        if left_val is not False and left_val is not None:
            if isinstance(left_val, float) and left_val == 0: return left_val
            if isinstance(left_val, str) and left_val == "": return left_val
            return left_val
        return self.right.eval(ctx)

@dataclass
class UnaryOp(Expr):
    """Uma operação prefixa com um operando."""
    operand: Expr
    op: Callable[[Value], Value]

    def eval(self, ctx: Ctx):
        value = self.operand.eval(ctx)
        return self.op(value)

@dataclass
class Call(Expr):
    """Uma chamada de função."""
    callee: Expr
    params: list[Expr]
    
    def eval(self, ctx: Ctx):
        func = self.callee.eval(ctx)
        args = [param.eval(ctx) for param in self.params]
        if callable(func):
            return func(*args)
        raise TypeError(f"'{func}' não é uma função!")

@dataclass
class This(Expr):
    """Acesso ao `this`."""
    # Campo vazio para satisfazer o sistema de nós
    _dummy: None = field(default=None, init=False)
    
    def eval(self, ctx: Ctx):
        try:
            return ctx["this"]
        except KeyError:
            raise NameError(f"variável this não existe!")

    # Validação Semântica para This
    def validate_self(self, cursor: Cursor):
        """Verifica se this está dentro de um método de classe."""
        if not cursor.is_scoped_to(Class):
            raise SemanticError(
                "Can't use 'this' outside of a class.",
                token="this"
            )

@dataclass
class Super(Expr):
    """Acesso a método ou atributo da superclasse."""
    name: str

    def eval(self, ctx: Ctx):
        try:
            # Procura a instância atual (this) no contexto
            instance = ctx['this']
            # Procura a superclasse no contexto
            superclass = ctx['super']
            # Busca o método na superclasse
            method = superclass.get_method(self.name)
            # Retorna o método ligado à instância
            return method.bind(instance)
        except KeyError as e:
            if 'this' in str(e):
                raise NameError(f"variável this não existe!")
            elif 'super' in str(e):
                raise NameError(f"variável super não existe!")
            else:
                raise

    # Validação Semântica para Super
    def validate_self(self, cursor: Cursor):
        """Verifica se super está dentro de um método de classe que tem superclasse."""
        # Primeiro verifica se está dentro de uma classe
        if not cursor.is_scoped_to(Class):
            raise SemanticError(
                "Can't use 'super' outside of a class.",
                token="super"
            )
        
        # Depois verifica se a classe tem superclasse
        try:
            class_cursor = cursor.class_scope()
            class_node = class_cursor.node
            if class_node.base is None:
                raise SemanticError(
                    "Can't use 'super' in a class with no superclass.",
                    token="super"
                )
        except ValueError:
            # Isso não deveria acontecer já que verificamos is_scoped_to(Class) acima
            raise SemanticError(
                "Can't use 'super' outside of a class.",
                token="super"
            )

@dataclass
class Assign(Expr):
    """Atribuição de variável."""
    name: str
    value: Expr

    def eval(self, ctx: Ctx):
        result = self.value.eval(ctx)
        ctx.assign(self.name, result)
        return result

@dataclass
class Getattr(Expr):
    """Acesso a atributo de um objeto."""
    obj: Expr
    name: str

    def eval(self, ctx: Ctx):
        obj_value = self.obj.eval(ctx)
        # Se for uma instância de LoxClass, buscar método corretamente
        from .runtime import LoxClass
        if isinstance(obj_value, LoxClass):
            try:
                return obj_value.get_method(self.name)
            except Exception:
                raise AttributeError(f"O objeto {obj_value} não possui o atributo '{self.name}'")
        try:
            return getattr(obj_value, self.name)
        except AttributeError:
            raise AttributeError(f"O objeto {obj_value} não possui o atributo '{self.name}'")

@dataclass
class Setattr(Expr):
    """Atribuição de atributo de um objeto."""
    obj: Expr
    name: str
    value: Expr

    def eval(self, ctx: Ctx):
        obj_val = self.obj.eval(ctx)
        val = self.value.eval(ctx)
        
        # Bloqueia setattr em classes e funções Lox
        from .runtime import LoxClass, LoxFunction
        if isinstance(obj_val, (LoxClass, LoxFunction)):
            raise LoxError("Apenas instâncias podem ter campos.")
        
        # Para LoxInstance, usa o método set_field
        from .runtime import LoxInstance
        if isinstance(obj_val, LoxInstance):
            obj_val.set_field(self.name, val)
        else:
            # Para outros objetos Python, usa setattr normal
            setattr(obj_val, self.name, val)
        
        return val

# COMANDOS

@dataclass
class Print(Stmt):
    """Representa uma instrução de impressão."""
    expr: Expr

    def eval(self, ctx: Ctx):
        value = self.expr.eval(ctx)
        lox_print(value)

@dataclass
class Return(Stmt):
    """Representa uma instrução de retorno."""
    value: Expr | None

    def eval(self, ctx: Ctx):
        return_value = self.value.eval(ctx) if self.value else None
        raise LoxReturn(return_value)

    # Validação Semântica para Return
    def validate_self(self, cursor: Cursor):
        """Verifica se return está dentro de uma função."""
        if not cursor.is_scoped_to(Function):
            raise SemanticError(
                "Can't return from top-level code.",
                token="return"
            )

@dataclass
class VarDef(Stmt):
    """Representa uma declaração de variável."""
    name: str
    initializer: Expr

    def eval(self, ctx: Ctx):
        value = self.initializer.eval(ctx)
        ctx.var_def(self.name, value)

    # Validação Semântica para VarDef
    def validate_self(self, cursor: Cursor):
        """Verifica se o nome da variável não é uma palavra reservada."""
        if self.name in RESERVED_WORDS:
            raise SemanticError(
                f"Cannot use reserved word '{self.name}' as a variable name.",
                token=self.name
            )

@dataclass
class If(Stmt):
    """Representa uma instrução condicional."""
    condition: Expr
    then_branch: Stmt
    else_branch: Stmt

    def eval(self, ctx: Ctx):
        condition_val = self.condition.eval(ctx)
        if condition_val is not False and condition_val is not None:
            self.then_branch.eval(ctx)
        else:
            self.else_branch.eval(ctx)

@dataclass
class While(Stmt):
    """Representa um laço de repetição."""
    condition: Expr
    body: Stmt

    def eval(self, ctx: Ctx):
        while True:
            condition_val = self.condition.eval(ctx)
            if condition_val is False or condition_val is None:
                break
            self.body.eval(ctx)

@dataclass
class Block(Node):
    """Representa um bloco de comandos."""
    stmts: list[Stmt]

    def eval(self, ctx: Ctx):
        new_ctx = ctx.push({})
        for stmt in self.stmts:
            stmt.eval(new_ctx)

    # Validação Semântica para Block
    def validate_self(self, cursor: Cursor):
        """Verifica se não há duas variáveis com o mesmo nome no mesmo bloco."""
        seen = set()
        for stmt in self.stmts:
            if isinstance(stmt, VarDef):
                if stmt.name in seen:
                    raise SemanticError(
                        f"Variable '{stmt.name}' has already been declared in this block.",
                        token=stmt.name
                    )
                seen.add(stmt.name)

@dataclass
class Function(Stmt):
    """Representa uma declaração de função."""
    name: str
    params: list[Var]
    body: Block

    def eval(self, ctx: Ctx):
        param_names = [p.name for p in self.params]
        function = LoxFunction(self.name, param_names, self.body, ctx)
        ctx.var_def(self.name, function)
        return None

    # Validação Semântica para Function
    def validate_self(self, cursor: Cursor):
        """
        Valida a declaração da função para:
        0. Nomes de parâmetros que são palavras reservadas. (NOVA VERIFICAÇÃO)
        1. Parâmetros com nomes duplicados.
        2. Variáveis no corpo que sombreiam parâmetros.
        """
        # 0. (NOVO) Verifica se algum parâmetro usa uma palavra reservada
        for param in self.params:
            if param.name in RESERVED_WORDS:
                raise SemanticError(
                    f"Cannot use reserved word '{param.name}' as a parameter name.",
                    token=param.name
                )
        
        # 1. Verifica parâmetros duplicados
        param_names = set()
        for param in self.params:
            if param.name in param_names:
                raise SemanticError(
                    f"Duplicate parameter name '{param.name}' in function declaration.",
                    token=param.name
                )
            param_names.add(param.name)

        # 2. Verifica se uma variável declarada no corpo tem o mesmo nome de um parâmetro
        for stmt in self.body.stmts:
            if isinstance(stmt, VarDef):
                if stmt.name in param_names:
                    raise SemanticError(
                        f"Variable '{stmt.name}' shadows a function parameter.",
                        token=stmt.name
                    )

@dataclass
class Class(Stmt):
    """Representa uma declaração de classe no código."""
    name: str
    methods: list["Function"]
    base: Optional[Var]

    def eval(self, ctx: Ctx):
        # Carrega a superclasse, caso exista
        superclass = None
        if self.base is not None:
            base_value = self.base.eval(ctx)
            if not isinstance(base_value, LoxClass):
                raise LoxError(f"'{self.base.name}' não é uma classe")
            superclass = base_value
        
        # Criamos um escopo para os métodos possivelmente diferente do escopo 
        # onde a classe está declarada
        if superclass is None:
            method_ctx = ctx
        else:
            method_ctx = ctx.push({"super": superclass})
        
        # Avaliamos cada método
        methods = {}
        for method in self.methods:
            method_name = method.name
            method_body = method.body
            method_args = [p.name for p in method.params]
            # Usa o contexto correto que pode incluir 'super'
            method_impl = LoxFunction(method_name, method_args, method_body, method_ctx)
            methods[method_name] = method_impl

        # Cria a classe e a define no contexto
        lox_class = LoxClass(self.name, methods, superclass)
        ctx.var_def(self.name, lox_class)
        return lox_class