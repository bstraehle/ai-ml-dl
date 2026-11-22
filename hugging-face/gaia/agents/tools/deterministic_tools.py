# References:

# https://docs.crewai.com/introduction

from crewai.tools import tool

class DeterministicTools():
    # Arithmetic

    @tool("Add Tool")
    def add_tool(a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a (float): First number
            b (float): Second number

        Returns:
            number: Result
        """
        print(f"🤖 DeterministicTools: add_tool: a={a}, b={b}")

        result = a + b
        print(f"🤖 DeterministicTools: add_tool: result={result}")
        return result
    
    @tool("Subtract Tool")
    def subtract_tool(a: float, b: float) -> float:
        """Subtract two numbers.
        
        Args:
            a (float): First number
            b (float): Second number

        Returns:
            number: Result
        """
        print(f"🤖 DeterministicTools: subtract_tool: a={a}, b={b}")

        result = a - b
        print(f"🤖 DeterministicTools: subtract_tool: result={result}")
        return result
    
    @tool("Multiply Tool")
    def multiply_tool(a: float, b: float) -> float:
        """Multiply two numbers.
        Args:
            a (float): First number
            b (float): Second number

        Returns:
            number: Result
        """
        print(f"🤖 DeterministicTools: multiply_tool: a={a}, b={b}")

        result = a * b
        print(f"🤖 DeterministicTools: multiply_tool: result={result}")
        return result
    
    @tool("Divide Tool")
    def divide_tool(a: float, b: float) -> float:
        """Divide two numbers.
        
        Args:
            a (float): First number
            b (float): Second number

        Returns:
            number: Result

        Raises:
            RuntimeError: If processing fails
        """
        print(f"🤖 DeterministicTools: divide_tool: a={a}, b={b}")

        if b == 0:
            raise RuntimeError("Cannot divide by zero.")

        result = a / b
        print(f"🤖 DeterministicTools: divide_tool: result={result}")
        return result
    
    @tool("Modulus Tool")
    def modulus_tool(a: float, b: float) -> float:
        """Get the modulus of two numbers.
        
        Args:
            a (float): First number
            b (float): Second number

        Returns:
            number: Result
        """
        print(f"🤖 DeterministicTools: modulus_tool: a={a}, b={b}")

        result = a % b
        print(f"🤖 DeterministicTools: modulus_tool: result={result}")
        return result