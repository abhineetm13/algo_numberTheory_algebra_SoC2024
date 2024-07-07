import random
# from typing import Self

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
    length: int = len(bin(x)[2:])
    k: int = (length-1)//2
    sqrt: int = 1<<k

    for i in range(k-1, -1, -1):
        if pow((sqrt + (1<<i)), 2) <= x:
            sqrt += 1<<i

    return sqrt

# print(floor_sqrt(123456))


def floor_root(x: int, p: int)->int:
    length: int = len(bin(x)[2:])
    k: int = (length-1)//p
    root: int = 1<<k

    for i in range(k-1, -1, -1):
        if pow((root + (1<<i)), p) <= x:
            root += 1<<i

    return root

# print(floor_root(1234, 10))

def is_perfect_power(x: int)->bool:
    length = len(bin(x)[2:])
    pow_range = length-1
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
    k = 0
    if 100 < n: k = 100
    else: k = n-1

    for i in range(0, k):
        a = random.randint(1, n-1)
        if is_in_miller_rabin_set(n, a) == False:
            return False
    
    return True

# print(is_prime(8683317618811886495518194401279999999))
# print(pow(1234, 123, 123456))


def gen_prime(m: int)->int:
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
    for i in range(2, floor_sqrt(n)+1):
        if n%i == 0: return i


def factor(n: int)->list[tuple[int, int]]:
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
    phi = 1
    factors = factor(n)
    for p, a in factors:
        phi = phi*pow(p, a-1)*(p-1)

    return phi

# print(euler_phi(1111111))


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
    

# """
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
        
# """
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