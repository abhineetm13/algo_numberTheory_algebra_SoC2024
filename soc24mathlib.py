def pair_gcd(a: int, b: int)->int:
    """Returns the gcd of two integers.
    
    Args:
        a: the first integer.
        b: the second integer.

    Returns:
        int: gcd of a and b.

    The Euclidean algorithm is used in the implementation.

    """
    if b == 0:
        return a
    else:
        return abs(pair_gcd(b, a%b))  # ??? Is abs needed? Yes

# print(pair_gcd(1, -1))


def pair_egcd_recursive(a: int, b: int)->tuple[int, int, int, int]:
    """This recursive function is used in pair_egcd function.
    
    Returns (a, b, x, y), where a, b are the arguments, and x, y are integers such that ax+by=gcd(a, b).
    
    Args:
        a: the first integer.
        b: the second integer.

    Returns:
        tuple: (a, b, x, y), where
            a and b are the arguments.
            x and y are integers such that ax+by=gcd(a, b).
    
    """
    if b == 0:
        return (a, b, 1, 0)
    else:
        next_value: tuple[int, int, int, int] = pair_egcd_recursive(b, a%b)
        return (a, b, next_value[3], (next_value[2]-(a//b)*next_value[3]))


def pair_egcd(a: int, b: int)->tuple[int, int, int]:
    """Returns (x, y, d) where d is gcd(a, b) and x, y are integers such that ax+by=d.

    Args:
        a: the first integer.
        b: the second integer

    Returns:
        tuple: (x, y, d) where
            d: gcd of a, b
            x, y are integers such that ax+by=d.

    The extended euclidean algorithm is implemented in this function.

    """
    value: int = pair_egcd_recursive(a, b)
    return (value[2], value[3], a*value[2]+b*value[3])


def gcd(*args: int)->int:
    """Returns the gcd of any number of integers.

    Args:
        *args: each argument is an integer.

    Returns:
        int: the gcd of all the arguments.
    
    """
    gcd: int = 0
    for arg in args:
        gcd = pair_gcd(arg, gcd)
    return gcd


def pair_lcm(a: int, b: int)->int:
    """Returns the lcm of two integers.

    Args:
        a: the first integer.
        b: the second integer.

    Returns:
        int: the lcm of a, b.

    The relation between gcd and lcm is used here.    

    """
    d: int = pair_gcd(a, b)
    return a*b//d

# print(pair_lcm(540, -54))


def lcm(*args: int)->int:
    """Returns the lcm of any number of integers.

    Args:
        *args: each argument is an integer.

    Returns:
        int: the lcm of all the integers.

    """
    lcm: int = 1 
    for arg in args:
        lcm = pair_lcm(lcm, arg)
    return lcm

# print(lcm(1, 2, 3))


def are_relatively_prime(a: int, b: int)->bool:
    """This functions checks if two numbers are coprime or not.

    Args:
        a: the first integer.
        b: the second integer.

    Returns:
        bool: True if a, b are coprime, False otherwise.

    The function checks if the gcd of the integers is 1 or not.
    
    """
    if pair_gcd(a, b) == 1:
        return True
    else:
        return False
    
# print(are_relatively_prime(548, 50))


def mod_inv(a: int, n: int)->int:
    """Returns the modular inverse of a modulo n.

    Args:
        a: the integer whose inverse is to be found.
        n: the inverse is to be found modulo n.

    Returns:
        int: inverse of a modulo n.

    Raises:
        ValueError: 
            This error is raised if inverse of a modulo n does not exist.
    
    """
    if pair_gcd(a, n) != 1:
        raise ValueError('Inverse does not exist')
        return None
    elif n == 1:
        return 0
    else:
        value: int = pair_egcd(a, n)
        return value[0]%n
            
# print(mod_inv(549, 50))


def crt(a: list[int], n: list[int])->int:
    """Solves a system of linear congruences of the form a=a[i] (mod n[i]).

    Returns the unique value of (a modulo product of all n[i]) such that a=a[i] (mod n[i]).

    Args:
        a: list of a[i], the residues modulo n[i]
        n: list of n[i], all n[i] should be pairwise coprime.

    Returns:
        int: the unique value of (a modulo product of all n[i]) such that a=a[i] (mod n[i]).

    The Chinese Remainder Theorem is applied here.
    
    """
    prod_n: int = 1
    for i in n:
        prod_n *= i
    ans: int = 0
    for item in zip(a, n):
        except_item: int = prod_n//item[1]
        t: int = mod_inv(except_item, item[1])
        e: int = t*except_item
        ans += item[0]*e
        ans = ans%prod_n
    return ans

# print(crt([1, 1, 2], [2, 3, 5]))


def is_quadratic_residue_prime(a: int, p: int)->int:
    """Checks whether integer a is a quadratic residue modulo prime p.

    Args:
        a: the integer to be checked.
        p: a prime number.

    Returns:
        int:
            1 if a is a quadratic residue modulo p,
            -1 if a is a quadratic non-residue modulo p,
            0 if a is not coprime to p.

    Euler's criterion is used in this function.
    
    """
    if p == 2:
        if a%2 == 1:
            return 1
        else:
            return 0
    criterion: int = pow(a, (p-1)//2, p)
    if criterion == 1:
        return 1
    elif criterion == p-1:
        return -1
    else:
        return 0
    
# print(is_quadratic_residue_prime(10111111111111111, 2))
    

def is_quadratic_residue_prime_power(a: int, p: int, e: int)->int:
    """Checks whether integer a is a quadratic residue modulo p^e.

    Args:
        a: the integer to be checked.
        p: a prime number.
        e: the power to which p is to be raised. (e>=1)

    Returns:
        1 if a is a quadratic residue modulo p^e.
        -1 if a is not a quadratic residue modulo p^e.
        0 if a is not coprime to p^e.

    Euler's criterion is used in this function.
    
    """
    if p == 2:
        if a%2 == 1:
            return 1
        else:
            return 0
    euler_phi: int = pow(p, e-1)*(p-1)
    criterion: int = pow(a, euler_phi//2, p)
    if criterion == 1:
        return 1
    elif criterion == p-1:
        return -1
    else:
        return 0
    
# print(is_quadratic_residue_prime_power(1111000011, 100, 2))
