import random
# from typing import Self
import fractions

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
    value: tuple[int, int, int, int] = pair_egcd_recursive(a, b)
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
    elif n == 1:
        return 0
    else:
        value: tuple[int, int, int] = pair_egcd(a, n)
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
        int: 
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


def floor_sqrt(x: int)->int:
    """ Returns the floor of the square root of x.

    Args:
        x: the input non-negative integer, whose sqrt is to be found.

    Returns:
        int: the floor of the square root of x.
    
    """
    length: int = len(bin(x)[2:])
    k: int = (length-1)//2
    sqrt: int = 1<<k

    for i in range(k-1, -1, -1):
        if pow((sqrt + (1<<i)), 2) <= x:
            sqrt += 1<<i

    return sqrt

# print(floor_sqrt(123456))


def floor_root(x: int, p: int)->int:
    """ Returns the floor of the p'th root of x.

    Args:
        x: the non-negative integer, whose root is to be found.
        p: a positive integer, such that floor of x^(1/p) is returned.

    Returns:
        int: floor of x^(1/p).
    
    """
    length: int = len(bin(x)[2:])
    k: int = (length-1)//p
    root: int = 1<<k

    for i in range(k-1, -1, -1):
        if pow((root + (1<<i)), p) <= x:
            root += 1<<i

    return root

# print(floor_root(1234, 10))

def is_perfect_power(x: int)->bool:
    """ Checks if x is a perfect power of an integer or not.

    Args:
        x: an integer

    Returns:
        bool: True if x = root^p for some integers root and p, False otherwise
    """
    length: int = len(bin(x)[2:])
    pow_range: int = length-1
    for p in range(2, pow_range+1):
        k: int = (length-1)//p
        root: int = 1<<k

        for i in range(k-1, -1, -1):
            if pow((root + (1<<i)), p) <= x:
                root += 1<<i

        if pow(root, p) == x: return True
    
    return False
    
# print(is_perfect_power(127))


def is_in_miller_rabin_set(n: int, a: int)->bool:
    """ Helper function used in is_prime function, used to implement Miller-Rabin algorithm
    
    """
    t = n-1
    h = 0
    while t%2 == 0:
        t = t//2
        h+=1
    b = pow(a, t, n)
    if b == 1: return True
    for j in range(0, h):
        if b == n-1 or b == -1 : return True
        if b == 1: return False
        b = pow(b, 2, n)
    return False


def is_prime(n: int)->bool:
    """ Checks if n is prime or not.

    Args:
        n: a positive integer

    Returns:
        bool: True if n is prime, False otherwise.
    
    The function uses Miller-Rabin algorithm.
    """
    k = 100

    for i in range(0, k):
        a = random.randint(1, n-1)
        if is_in_miller_rabin_set(n, a) == False:
            return False
    
    return True

# print(is_prime(8683317618811886495518194401279999999))
# print(pow(1234, 123, 123456))


def gen_prime(m: int)->int:
    """ Generates a random prime less than or equal to m.

    Args:
        m: the upper bound for the random prime

    Returns:
        int: a random prime in the range [2, m]
    
    This function uses the Niller-Rabin algorithm
    """
    while True:
        n = random.randint(2, m)

        is_prime_trial = True
        for p in [2, 3, 5, 7]:
            if n == p: return n
            if n%p == 0:
                is_prime_trial = False
                break

        if is_prime_trial:
            if is_prime(n):
                return n


# x = gen_prime(8683317618811886495518194401279999999)
# print(x, is_prime(x))


def gen_k_bit_prime(k: int)->int:
    """ Generates a random prime of k bits

    Args:
        k: a positive integer

    Returns:
        int: a random k-bit prime
    
    This function uses the Miller-Rabin algorithm
    """
    while True:
        n = random.randint(pow(2, k-1), pow(2, k)-1)
        
        is_prime_trial = True
        for p in [2, 3, 5, 7]:
            if n == p: return n
            if n%p == 0:
                is_prime_trial = False
                break

        if is_prime_trial:
            if is_prime(n):
                return n

# x = gen_k_bit_prime(868)
# print(x, is_prime(x))


def find_factor(n: int):
    """ Helper function for factor function, it finds a factor of n.
    
    """
    for i in range(2, floor_sqrt(n)+1):
        if n%i == 0: return i


def factor(n: int)->list[tuple[int, int]]:
    """ Returns the prime factorisation of n.

    Args:
        n: a positive integer.

    Returns:
        list[tuple[int, int]]: Each tuple in the list is of the form (p, e), where p is a prime which divides n and e is the exponent of p in n.
            The list is sorted in ascending order of the first element of each tuple.
    
    """
    if n == 1: return []

    if is_prime(n):
        return [(n, 1)]
    
    d = find_factor(n)
    # print(d, n//d)
    n_factorisation = dict(factor(n//d))

    new_factors = dict(factor(d))
    for p in new_factors:
        if p in n_factorisation:
            n_factorisation[p] += new_factors[p]
        else:
            n_factorisation[p] = new_factors[p]

    factors = []
    for p in sorted(n_factorisation):
        factors.append((p, n_factorisation[p]))

    return factors

# print(factor(20000000))


def euler_phi(n: int)->int:
    """ Returns the value of euler's totient function, phi, of n.

    Args:
        n: a positive integer, whose phi needs to be found.

    Returns:
        int: phi(n)
    
    """
    phi = 1
    factors = factor(n)
    for p, a in factors:
        phi = phi*pow(p, a-1)*(p-1)

    return phi

# print(euler_phi(1111111))


def gaussian_elimination(A: list[list[int]])->list[list[fractions.Fraction]]:
    """ Performs gaussian elimination on a matrix of integers

    Args:
        A: input matrix.

    Returns:
        list[list[fractions.Fraction]]: the echelon form of A, which is a matrix of fractions. 
    
    """
    """
        A is m*n, each element of A is a row
        m = no. of rows, n = no. of columns
        A[i-1] is ith row, A[:][j-1] is jth column
    """
    B = [[fractions.Fraction(A[i][j], 1) for j in range(len(A[i]))] 
                for i in range(len(A))]
    r = 0
    m = len(A)
    n = len(A[0])
    for j in range(1, n+1):
        l = 0
        i = r
        while l == 0 and i < m:
            i += 1
            # print(i)
            if B[i-1][j-1] != 0: l = i
        if l != 0:
            r = r+1
            swap = B[r-1][:]
            B[r-1] = B[l-1][:]
            B[l-1] = swap[:]
            b = 1/B[r-1][j-1]
            for iter in range(len(B[r-1])): B[r-1][iter] *= b
            for i in range(1, m+1):
                if i != r:
                    b = B[i-1][j-1]
                    for iter in range(len(B[i-1])): B[i-1][iter] -= b*B[r-1][iter]            

    return B


class QuotientPolynomialRing:
    pi_generator: list[int]
    element: list[int]

    def __init__(self, poly: list[int], pi_gen:list[int])->None:
        if pi_gen == []:
            raise ValueError('Empty pi_generator')
        if pi_gen[len(pi_gen)-1] != 1:
            raise ValueError('Pi_generator is not monic')
        self.pi_generator = pi_gen
        self.element = poly
        return

    @staticmethod
    def Deg(p: list[int])->list[int]:
        if len(p) == 0: return 0
        for i in range(len(p)-1, -1, -1):
            if p[i]!= 0: return i
        return 0

    @staticmethod
    def Mod_gen(p: list[int], gen: list[int])->"QuotientPolynomialRing":
        while len(p) >= len(gen):
            order_diff = len(p)-len(gen)
            for i in range(len(p)-order_diff):
                p[i+order_diff] -= p[len(p)-1]*gen[i]
            p.pop()
        return QuotientPolynomialRing(p, gen)

    @staticmethod
    def Add(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        
        sum = []

        l1 = min(len(poly1.element), len(poly2.element))
        l2 = max(len(poly1.element), len(poly2.element))
        Max = poly1
        if len(poly1.element) == l2: Max = poly1
        else: Max = poly2

        for i in range(l1):
            sum.append(poly1.element[i]+poly2.element[i])

        for i in range(l1, l2):
            sum.append(Max.element[i])

        sum = QuotientPolynomialRing.Mod_gen(sum, poly1.pi_generator)
        
        return sum
    
    @staticmethod
    def Sub(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        
        e1: list = poly1.element
        e2: list = poly2.element
        l: int = 0
        longer: list = []
        if len(e1) > len(e2): 
            l = len(e2)
            longer = e1
        else: 
            l = len(e1)
            longer = e2
        diff: list = []
        for i in range(l):
            diff.append(e1[i]-e2[i])

        if longer == e1:
                for i in range(l, len(longer)):
                    diff.append(longer[i])
        else: 
            for i in range(l, len(longer)):
                diff.append(-longer[i])

        diff = QuotientPolynomialRing.Mod_gen(diff, poly1.pi_generator)

        return diff
    
    @staticmethod
    def Mul(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        deg = 2*(len(poly1.element)-1)
        prod = [0 for x in range(deg+1)]
        for i in range(len(poly1.element)):
            for j in range(len(poly2.element)):
                prod[i+j] += poly1.element[i]*poly2.element[j]

        prod = QuotientPolynomialRing.Mod_gen(prod, poly1.pi_generator)

        return prod
    
    @staticmethod
    def Scale(p: list[int], a: int)->list[int]:
        for i in range(len(p)):
            p[i] *= a
        return p
    
    @staticmethod
    def Common_factor(p: list[int])->int:
        f = p[QuotientPolynomialRing.Deg(p)]
        if f == 0: return 0
        for i in range(len(p)):
            if p[i] != 0: f = pair_gcd(f, p[i])
        return f
    
    @staticmethod
    def Reduce(p: list[int])->list[int]:
        l = len(p)
        gcd = 0
        for i in range(l):
            if p[i] != 0:
                gcd = p[i]
                break
        if gcd == 0: return p
        for i in range(l):
            if p[i] != 0: gcd = pair_gcd(gcd, p[i])
        for i in range(l-1, -1, -1):
            if p[i] > 0: break
            if p[i] < 0:
                gcd*=-1
                break
        for i in range(l):
            p[i] = p[i]//gcd
        return p

    @staticmethod
    def GCD(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Different pi_generators")
        a = QuotientPolynomialRing.Common_factor(poly1.element)
        b = QuotientPolynomialRing.Common_factor(poly2.element)
        # print(a, b)
        f = pair_gcd(a, b)
        p1 = poly1.element  # Assuming neither p1 nor y are zero polynomials
        p2 = poly2.element
        pow_X_1 = 0
        while p1[0] == 0:
            for i in range(len(p1)-1): p1[i] = p1[i+1]
            p1[len(p1)-1] = 0
            pow_X_1 += 1
        pow_X_2 = 0
        while p2[0] == 0:
            for i in range(len(p2)-1): p2[i] = p2[i+1]
            p2[len(p2)-1] = 0
            pow_X_2 += 1
        # return min(pow_X_1, pow_X_2)

        x = QuotientPolynomialRing.Reduce(p1)
        y = QuotientPolynomialRing.Reduce(p2)
        zeroes = [0 for i in range(len(x))] # Assuming len(x) == len(y)
        while True:
            if y == zeroes: 
                # print(x)
                break
            while x[0] == 0:
                for i in range(len(x)-1): x[i] = x[i+1]
                x[len(x)-1] = 0
            while y[0] == 0:
                for i in range(len(y)-1): y[i] = y[i+1]
                y[len(y)-1] = 0
            # print(QuotientPolynomialRing.Deg(x), QuotientPolynomialRing.Deg(y))
            if QuotientPolynomialRing.Deg(x) > QuotientPolynomialRing.Deg(y):
                swap = x; x = y; y = swap
                # print(1)
        
            # print(x, y)
            if y[0] % x[0] == 0:
                c = y[0]//x[0]
                for i in range(len(x)): y[i] -= x[i]*c
            elif y[0] == 1: 
                y = QuotientPolynomialRing.Scale(y, x[0])
                for i in range(len(x)): y[i] -= x[i]
                # print(y)
            elif y[0] == -1:
                y = QuotientPolynomialRing.Scale(y, x[0])
                for i in range(len(x)): y[i] += x[i]
            else:
                # print(x[0], y[0])
                a, b, d = pair_egcd(x[0], y[0])
                # print(a, b, d)
                x = QuotientPolynomialRing.Scale(x, a); y1 = QuotientPolynomialRing.Scale(y, b)
                for i in range(len(x)): x[i] += y1[i]
                for i in range(len(x)): y[i] -= x[i]*(y[0]//x[0])
                # print(x[0], y[0])
            x = QuotientPolynomialRing.Reduce(x)
            y = QuotientPolynomialRing.Reduce(y)
            # print(x, y)
            # print()
    
        x = QuotientPolynomialRing.Scale(x, f)
        d = min(pow_X_1, pow_X_2)
        for p in range(d):
            for i in range(len(x)-1, -1, -1): x[i] = x[i-1] 
            x[0] = 0
        # print(x)
        return QuotientPolynomialRing(x, poly1.pi_generator)
    
    @staticmethod
    def Inv(poly: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        p = poly.element[:]
        base = poly.pi_generator[:]
        d1 =QuotientPolynomialRing.Deg(p)
        d2 = QuotientPolynomialRing.Deg(base)
        
        # Creating matrix for solving equations
        A = []
        for iter1 in range(d1+d2):
            # print(1, iter1)
            Ai = [0 for i in range(d1+d2)]
            for iter2 in range(d2):
                # print(2, iter2)
                if iter1-iter2 > d1: Ai[iter2] = 0
                elif iter2 <= iter1: Ai[iter2] = p[iter1-iter2]
            for iter2 in range(d2, d1+d2):
                if iter1-iter2 > 0: Ai[iter2] = 0
                elif iter2-d2 <= iter1: Ai[iter2] = -base[iter1+d2-iter2]
            A.append(Ai)
        
        A[0].append(1)
        for i in range(1, len(A)):
            A[i].append(0)
        # print(A)

        # Solving the equations
        B = gaussian_elimination(A)
        # print(B)

        inv = [0 for i in range(d2)]
        null = [fractions.Fraction(0, 1) for i in range(d1+d2+1)]
        for iter in range(d1+d2):
            if B[iter] == null: raise ValueError("Inverse does not exist")
            if B[iter][d1+d2].denominator != 1: raise ValueError("Inverse does not exist")  
        for iter in range(d2):
            inv[iter] = int(B[iter][d1+d2])
        # print(inv)
        return QuotientPolynomialRing(inv, base)
"""
    @staticmethod
    def Inv(poly: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        a = poly.element
        r = QuotientPolynomialRing.Deg(a)
        c = poly.pi_generator
        n = QuotientPolynomialRing.Deg(c)
        b = []
        d = 1
        if c[0] == 0:
            if a[0] == 1: b.append(1)
            elif a[0] == -1: b.append(-1)
            else: raise ValueError("Inverse does not exist")
        else:
            if pair_gcd(a[0], c[0]) != 1: raise ValueError("Inverse does not exist")
            b.append(mod_inv(a[0], c[0]))
        for i in range(1, n):
            d = 0
            for j in range(i):
                d -= a[i-j]*b[j]
            if d == 0: 
                b.append(0)
            elif c[i] == 0:
                if d%a[0] != 0: raise ValueError("Inverse does not exist")
                b.append(d//a[0])
            else:
                if d%pair_gcd(a[0], c[i]) != 0: raise ValueError("Inverse does not exist")
                d = d//pair_gcd(a[0], c[i])
                a1 = a[0]//pair_gcd(a[0], c[i])
                c1 = c[i]//pair_gcd(a[0], c[i])
                b.append(d*mod_inv(a1, c1))
            # print(b)

        # print(b)
        b_poly = QuotientPolynomialRing(b, c)
        unit = [0 for i in range(n)]; unit[0] = 1
        # print(unit)
        if QuotientPolynomialRing.Mul(poly, b_poly).element != unit:
            raise ValueError("Inverse does not exist")

        return b_poly
        
"""
############################################################################################
"""
    def Mod(poly1: list[int], poly2: list[int], length)->list[int]:
        el = poly1
        coeff = 0

        l = len(poly2)
        for i in range(len(poly2)-1, -1, -1):
            if poly2[i] != 0: 
                coeff = poly2[i]
                break
            l -= 1

        for i in range(len(el)):
            el[i] *= coeff

        while len(el) >= l:
            order_diff = len(el) - l
            m = el[len(el)-1]//coeff
            for i in range(l):
                el[i+order_diff] -= m*poly2[i]
            el.pop()

        # for i in range(len(el)):
        #     if el[i]%coeff == 0: el[i] = el[i]//coeff
        #     else: el[i] = el[i]/coeff

        l = len(el)
        for i in range(l, length):
            el.append(0)
            
        return el

    @staticmethod
    def GCD_rec(poly1: Self, poly2: Self)->Self:
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        
        if len(poly2.element) == 0: return poly1
        if poly2.element == [0 for x in range(len(poly2.element))]: return poly1

        mod = QuotientPolynomialRing.Mod(poly1.element, poly2.element, len(poly1.element))
        mod_poly = QuotientPolynomialRing(mod, poly1.pi_generator)
        return QuotientPolynomialRing.GCD(poly2, mod_poly)
    
    def GCD(poly1: Self, poly2: Self)->Self:
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        
        gcd = QuotientPolynomialRing.GCD_rec(poly1, poly2).element

        l = len(gcd)
        if l == 0: return gcd
        factor = 0
        for x in gcd:
            if x != 0:
                factor = x
                break
        for x in gcd:
            if x != 0: factor = pair_gcd(factor, x)

        leading_coeff = 0
        for i in range(len(gcd)-1, -1, -1):
            if gcd[i] != 0: 
                leading_coeff = gcd[i]
                break

        if leading_coeff < 0: factor *= -1
        for i in range(len(gcd)):
            gcd[i] = gcd[i]//factor

        return QuotientPolynomialRing(gcd, poly1.pi_generator)
        """
#####################################################################################

# print(QuotientPolynomialRing.Inv(QuotientPolynomialRing([1, 2, 1], [0, 3, 3, 1])).element)

x = QuotientPolynomialRing([1, 3, 3, 1], [0, 1, 1])
y = QuotientPolynomialRing([2, 4, 2], [0, 1, 1])

# print(QuotientPolynomialRing.Mod([3, 7, 5, 1], [0, 1, 1]))
# print(QuotientPolynomialRing.Sub(x, y).element)
# print(QuotientPolynomialRing.Add(x, y).element, QuotientPolynomialRing.Add(x, y).pi_generator)
# print(QuotientPolynomialRing.GCD(x, y).element)
# print(QuotientPolynomialRing.Inv(QuotientPolynomialRing([0, 0, 1, 0], [7, 0, 0, 3, 1])).element)

p1 = QuotientPolynomialRing([-3, -5, -1, 1], [7, 0, 0,  3, 1])
p2 = QuotientPolynomialRing([1, 5, 7, 3], [7, 0, 0,  3, 1])
# print(QuotientPolynomialRing.GCD(p1, p2).element)
# print(QuotientPolynomialRing.Mod([-3, -5, -1, 1], [1, 5, 7, 3], 4))


def mod_base(p: list[int], r: int)->list[int]:
        while len(p) >= r+1:
            order_diff = len(p)-(r+1)
            p[len(p)-r-1] += p[len(p)-1]
            p.pop()

        return p

def mul(p1: list[int], p2: list[int], r: int, n: int)->list[int]:
        # print("a")
        l1 = len(p1); l2 = len(p2)
        # print((l1-1)+(l2-1)+1)
        prod = [0 for i in range((l1-1)+(l2-1)+1)]
        for i in range(l1):
            for j in range(l2):
                prod[i+j] += (p1[i]*p2[j])%n
                prod[i+j] %= n
        prod = mod_base(prod, r)
        return prod

def fast_exp(p,power,r, n):
    # print(power)
    if power == 0: return [1]
    elif power%2 == 0:
        return fast_exp(mul(p, p, r, n), power//2, r, n)
    else:
        return mul(fast_exp(mul(p, p, r, n), (power-1)//2, r, n), p, r, n)


def aks_satisfied(j, r, n):
    # print(1)
    x = fast_exp([j, 1], n, r, n)
    # print(2)
    y = fast_exp([0, 1], n, r, n)
    
    l1 = len(x)
    l2 = len(y)

    if l1 >= l2:
        for i in range(l2):
            if (x[i]-y[i])%n != 0: return False
        for i in range(l2, l1):
            if x[i]%n != 0: return False
    else:
        for i in range(l1):
            if (x[i]-y[i])%n != 0: return False
        for i in range(l1, l2):
            if y[i]%n != 0: return False

    return True


def aks_test(n: int)->bool:
    if is_perfect_power(n): return False

    r = 2
    while True:
        if pair_gcd(n, r) != 1: break
        else:
            o = 1
            pow_n = n
            while pow_n%r != 1:
                pow_n*=n
                o+=1
            if o>4*pow(len(bin(n)[2:]), 2):
                break
        r+=1

    if r==n: return True
    if pair_gcd(n, r)>1: return False
    # print(r)
    for j in range(1, 2*len(bin(n)[2:])*floor_sqrt(r)+1):
        # print(j)
        if aks_satisfied(j, r, n): return False

    return True

# print(aks_test(7))

# print(factor(1111111))
# print(factor(123467))
# print(find_factor_pollard_rho(4, 2, 1))
# print(is_prime(3698849471))


def get_generator(p: int)->int:
    """ Returns a generator of (Z_p)^*, where p is a prime.

    Args:
        p: a prime number

    Returns:
        int: a generator of (Z_p)^*
    
    """
    factorisation: list[tuple[int, int]] = factor(p-1)
    r: int = len(factorisation)
    c: int = 1
    for i in range(0, r):
        b: int = 1
        a: int = 0
        while b == 1:
            a = random.randint(1, p-1)
            b = pow(a, (p-1)//factorisation[i][0], p)
            # print(b, end = " ")
        # print(a)
        c *= pow(a, (p-1)//pow(factorisation[i][0], factorisation[i][1]), p)
        c %= p
    return c


def discrete_log(x: int, g: int, p: int)->int:
    """ Returns the discrete logarithm of x to the base g in (Z_p)^*, where p is a prime

    Args:
        x: an element of (Z_p)^*
        g: an element of (Z_p)^*
        p: a prime number

    Returns:
        int: the discrete logarithm of x to the base g in (Z_p)^*

    Raises:
        ValueError: Raised if the discrete logarithm does not exist.
    
    The baby step/giant step method is used in this function
    
    """
    q = p-1
    m = floor_sqrt(q)
    values = {}
    value = 1
    for i in range(m+1):
        values[value] = i
        value *= g
        value %= p
    # print(values)
    value = x
    mul = pow(g, q-m, p)
    # print(mul)
    for i in range(m+1):
        # print(value)
        if value in values:
            floor = values[value]
            return (floor + i*m)%q
        value *= mul
        value %= p

    raise ValueError("Discrete logarithm does not exist.")


def legendre_symbol(a: int, p: int)->int:
    """ Returns the Legendre symbol (a | p), where p is a prime

    Args:
        a: an integer
        p: a prime number

    Returns:
        int: The Legendre symbol, (a | p)
    
    """
    if a%p == 0: return 0
    if p == 2: return 1
    if pow(a, (p-1)//2, p) == 1: return 1
    else: return -1


def jacobi_symbol(a: int, n: int)->int:
    """ Returns the Jacobi symbol (a | n)

    Args:
        a: an integer
        n: a positive integer

    Returns:
        int: The Jacobi symbol (a | n)
    
    """
    if pair_gcd(a, n) != 1: return 0
    j = 1
    factors = factor(n)
    # print(factors)
    for q, i in factors:
        # print(legendre_symbol(a, q))
        j *= pow(legendre_symbol(a, q), i)
    return j


def modular_sqrt_prime(x: int, p: int)->int:
    """ Returns the modular square root of a number modulo a prime number

    Args:
        x: the number 
        p: the prime number, where square root is found in Z_p

    Returns:
        int: modular square root of x modulo p
    
    """
    if is_quadratic_residue_prime(x, p) != 1:
        raise ValueError("Modular square root does not exist")
    if p == 2: return 1
    if p%4 == 3: 
        p1 = pow(x, (p+1)//4, p)
        if p1 > p-p1: return p-p1
        else: return p1

    c = 2
    while is_quadratic_residue_prime(c,p) != -1: c = random.randint(2, p-1)

    h = 0
    m = p-1
    while m%2 == 0:
        m = m//2
        h += 1
    c = pow(c, m, p)
    x1 = pow(x, m, p)
    a = discrete_log(x1, c, p)
    
    p1 = (pow(c, a//2, p)*pow(x, -(m//2), p))%p
    if p1 < p-p1: return p1
    else: return p-p1


def modular_sqrt_prime_power(x: int, p: int, e: int)->int: # assuming odd prime
    if x%pow(p, e) == 0: return 0
    if is_quadratic_residue_prime_power(x, p, e) != 1:
        raise ValueError("Modular square root does not exist")
    
    b = modular_sqrt_prime(x, p)
    for i in range(1, e):
        a1 = mod_inv(2*b, p)
        b1 = (x-b*b)//pow(p, i)
        h = (a1*b1)%p
        b = (b + h*pow(p, i))%pow(p, i+1)
    p1 = b
    if p1 > pow(p, e)-p1: return pow(p, e)-p1
    else: return p1


def modular_sqrt_2_pow(x: int, e: int)->list[int]:
    sqrts = []
    num = pow(2, e)
    if x%2 == 0:
        for r in range(0, num, 2):
            if (r*r-x)%num == 0: sqrts.append(r)
        if len(sqrts) == 0: raise ValueError("Modular square root does not exist")
    else:
        for r in range(1, num, 2):
            if (r*r-x)%num == 0: sqrts.append(r)
        if len(sqrts) == 0: raise ValueError("Modular square root does not exist")
    return sqrts

# print(modular_sqrt_2_pow(68, 7))


def modular_sqrt(x: int, z: int)->int:
    y = z
    pow_2 = 0
    while y%2 == 0:
        pow_2+=1
        y = y//2

    factors = factor(y)
    sq_roots = []

    for prime, power in factors:
        sq_roots.append(modular_sqrt_prime_power(x%pow(prime, power), prime, power))

    if pow_2 != 0:
        factors.append((2, pow_2))
        sqrts_2 = modular_sqrt_2_pow(x, pow_2)
        # print(sqrts_2)

    n = [pow(prime, power) for prime, power in factors]
    
    # For listing all possibilities of sq. roots modulo prime-powers
    if len(sq_roots) != 0:
        a_possible = [[sq_roots[0]], [-sq_roots[0]]] 
        for i in range(1, len(sq_roots)):
            t = len(a_possible)
            for j in range(t):
                a1 = a_possible[j][:]
                a_possible[j].append(sq_roots[i])
                a1.append(-sq_roots[i])
                a_possible.append(a1)

        if pow_2 != 0:
            t = len(a_possible)
            for i in range(t):
                a1 = a_possible[i][:]
                a_possible[i].append(sqrts_2[0])
                for j in range(1, len(sqrts_2)):
                    a2 = a1[:]
                    a2.append(sqrts_2[j])
                    a_possible.append(a2)
        # print(a_possible)

        roots = []
        for a in a_possible:
            roots.append(crt(a, n)%z)
        # print(roots)
        return min(roots)

    else:
        return min(r%z for r in sqrts_2)


def is_smooth(m: int, y: int)->bool:
    for i in range(2, y+1)  :
        if m%i == 0:
            while m%i == 0:
                m = m//i
        if m <= y: return True
    return False


def gaussian_elimination_modulo_p(A: list[list[int]], p: int)->list[list[int]]:
    """
        A is m*n, each element of A is a row
        m = no. of rows, n = no. of columns
        A[i-1] is ith row, A[:][j-1] is jth column
    """
    B = [[A[i][j] for j in range(len(A[i]))] 
                for i in range(len(A))]
    r = 0
    m = len(A)
    n = len(A[0])
    for j in range(1, n+1):
        l = 0
        i = r
        while l == 0 and i < m:
            i += 1
            # print(i)
            if B[i-1][j-1]%p != 0: l = i
        if l != 0:
            r = r+1
            swap = B[r-1][:]
            B[r-1] = B[l-1][:]
            B[l-1] = swap[:]
            b = mod_inv(B[r-1][j-1], p)
            for iter in range(len(B[r-1])): 
                B[r-1][iter] *= b
                B[r-1][iter] %= p
            for i in range(1, m+1):
                if i != r:
                    b = B[i-1][j-1]
                    for iter in range(len(B[i-1])): 
                        B[i-1][iter] -= b*B[r-1][iter]
                        B[i-1][iter] %= p            

    return B

def prob_find_factor(n: int)->int:
    y = pow(2, floor_sqrt((len(bin(n)[2:])*len(bin(len(bin(n)[2:]))[2:]))//2))
    primes = []
    for i in range(2, y+1):
        if is_prime(i): primes.append(i)
    k = len(primes)
    # print(y, primes, k)

    v = []
    a = []

    d = random.randint(1, n-1)
    while pair_gcd(d, n) != 1: d = random.randint(1, n-1)
    # print(d)

    counter_lim = 10000 # I am getting lucky with this limit, I need it to prevent an infinite loop

    i = 0
    while True:
        i = i+1
        # print(i)
        counter = 0
        while counter < counter_lim:
            counter+=1
            a_i = random.randint(1, n-1)
            while pair_gcd(a_i, n) != 1: a_i = random.randint(1, n-1)
            m_i = (a_i*a_i*d)%n
            # print(a_i, m_i)
            e_i = [0 for iter in range(k)]
            for iter in range(k):
                pow_p_i = 0
                while m_i%primes[iter] == 0:
                    m_i = m_i//primes[iter]
                    pow_p_i += 1
                e_i[iter] = pow_p_i
            if m_i == 1: 
                break
        if counter == counter_lim:
            # print("noooooooooooooo1")
            return prob_find_factor(n)
        
        v.append(e_i)
        v[i-1].append(1)
        a.append(a_i)
        if i == k+2: break
    
    # print(a)
    # print(v)

    # print(len(v), len(v[0]))

    A = [[v[i][j] for i in range(k+2)] for j in range(k+1)]
    G = gaussian_elimination_modulo_p(A, 2)

    # print(G)

    zeroes = [0 for i in range(len(A[0]))]
    not_possible = zeroes[:]; not_possible[k] = 1

    c = []
    for Gi in G:
        if Gi == not_possible:
            # print("nooooooooooooo2")
            return prob_find_factor(n)
        elif Gi == zeroes:
            c.append(0)
        else:
            c.append(1)
    
    if c == [0 for i in range(k+1)]: 
        # print("noooooooooooooooo3")
        return prob_find_factor(n)
    c.append(1)
    # print(c)

    alpha = 1

    for i in range(k+2):
        alpha *= pow(a[i], c[i])
    # print(alpha, d)

    e = [0 for i in range(k+1)]
    for i in range(k+1):
        for j in range(k+2):
            e[i] += c[j]*v[j][i]

    # print(e)

    beta = 1
    for i in range(k):
        beta *= pow(primes[i], e[i]//2)
    # beta //= pow(d, e[k]//2)

    # print(alpha, beta)

    gamma = (alpha//beta)*pow(d, e[k]//2)

    # print(gamma)
    if gamma%n == 1 or gamma%n == -1:
        # print("noooooooooooooooo4")
        return prob_find_factor(n)
    
    factor = pair_gcd((gamma-1)%n, n)
    if factor == 1: 
        # print("noooooooooooooooooooo5")
        return prob_find_factor(n)
    return factor

# print(prob_find_factor(1408198281))

def probabilistic_factor(n: int)->list[tuple[int, int]]:
    factors = []
    # to ensure that n is odd
    pow_2 = 0
    while n%2 == 0:
        pow_2 += 1
        n = n//2
    if pow_2 != 0: factors.append((2, pow_2))
    if n == 1: return factors

    # to ensure that n is not prime
    if is_prime(n):
        factors.append((n, 1))
        return factors
    
    # to ensure that n is not of the form p^e for some prime p
    length = len(bin(n)[2:])
    pow_range = length-1
    for p in range(2, pow_range+1):
        k: int = (length-1)//p
        root: int = 1<<k

        for i in range(k-1, -1, -1):
            if pow((root + (1<<i)), p) <= n:
                root += 1<<i

        if pow(root, p) == n:
            factors.append((root, p))
            return factors
        
    # To ensure that n is not y-smooth    
    y = pow(2, floor_sqrt((len(bin(n)[2:])*len(bin(len(bin(n)[2:]))[2:]))//2))
    for i in range(2, y+1):
        pow_i = 0
        while n%i == 0:
            pow_i += 1
            n = n//i
        if pow_i != 0: factors.append((i, pow_i))
    if n == 1: return factors

    # The actual algorithm

    if is_prime(n):
        factors.append((n, 1))
        return factors
    
    d = prob_find_factor(n)
    # print(d, n//d)
    n_factorization = dict(probabilistic_factor(n//d))
    new_factors = dict(probabilistic_factor(d))
    for prime in new_factors:
        if prime in n_factorization:
            n_factorization[prime] += new_factors[prime]
        else:
            n_factorization[prime] = new_factors[prime]

    dict_factors = dict(factors)
    for prime in n_factorization:
        if prime in dict_factors:
            dict_factors[prime] += n_factorization[prime]
        else:
            dict_factors[prime] = n_factorization[prime]

    factors_final = []
    for prime in sorted(dict_factors):
        factors_final.append((prime, dict_factors[prime]))

    return factors_final
