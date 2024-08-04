import random
import fractions

def pair_gcd(a: int, b: int) -> int:
    """Returns the gcd of two integers.
    
    Args:
        a: The first integer
        b: The second integer

    Returns:
        int: Gcd of a and b

    The Euclidean algorithm is used in the implementation.

    """
    if b == 0:
        return a
    else:
        return abs(pair_gcd(b, a%b))  # ??? Is abs needed? Yes

# print(pair_gcd(1, -1))


def pair_egcd_recursive(a: int, b: int) -> tuple[int, int, int, int]:
    """This recursive function is used in pair_egcd function.
    
    Returns (a, b, x, y), where a, b are the arguments, and x, y are integers such that ax+by=gcd(a, b).
    
    Args:
        a: The first integer
        b: The second integer

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


def pair_egcd(a: int, b: int) -> tuple[int, int, int]:
    """Returns (x, y, d) where d is gcd(a, b) and x, y are integers such that ax+by=d.

    Args:
        a: The first integer
        b: The second integer

    Returns:
        tuple: (x, y, d) where
            d: gcd of a, b
            x, y are integers such that ax+by=d

    The extended euclidean algorithm is implemented in this function.

    """
    value: tuple[int, int, int, int] = pair_egcd_recursive(a, b)
    return (value[2], value[3], a*value[2]+b*value[3])


def gcd(*args: int) -> int:
    """Returns the gcd of any number of integers

    Args:
        *args: Each argument is an integer

    Returns:
        int: The gcd of all the arguments
    
    """
    gcd: int = 0
    for arg in args:
        gcd = pair_gcd(arg, gcd)
    return gcd


def pair_lcm(a: int, b: int) -> int:
    """Returns the lcm of two integers.

    Args:
        a: The first integer
        b: The second integer

    Returns:
        int: The lcm of a, b

    The relation between gcd and lcm is used here.

    """
    d: int = pair_gcd(a, b)
    return a*b//d

# print(pair_lcm(540, -54))


def lcm(*args: int) -> int:
    """Returns the lcm of any number of integers.

    Args:
        *args: Each argument is an integer

    Returns:
        int: The lcm of all the integers

    """
    lcm: int = 1 
    for arg in args:
        lcm = pair_lcm(lcm, arg)
    return lcm

# print(lcm(1, 2, 3))


def are_relatively_prime(a: int, b: int) -> bool:
    """This functions checks if two numbers are coprime or not.

    Args:
        a: The first integer
        b: The second integer

    Returns:
        bool: True if a, b are coprime, False otherwise

    The function checks if the gcd of the integers is 1 or not.
    
    """
    if pair_gcd(a, b) == 1:
        return True
    else:
        return False
    
# print(are_relatively_prime(548, 50))


def mod_inv(a: int, n: int) -> int:
    """Returns the modular inverse of a modulo n.

    Args:
        a: The integer whose inverse is to be found.
        n: The inverse is to be found modulo n.

    Returns:
        int: Inverse of a modulo n.

    Raises:
        ValueError: 
            If inverse of a modulo n does not exist.
    
    """
    if pair_gcd(a, n) != 1:
        raise ValueError('Inverse does not exist')
    elif n == 1:
        return 0
    else:
        value: tuple[int, int, int] = pair_egcd(a, n)
        return value[0]%n
            
# print(mod_inv(549, 50))


def crt(a: list[int], n: list[int]) -> int:
    """Solves a system of linear congruences of the form a=a[i] (mod n[i]).

    Returns the unique value of (a modulo product of all n[i]) such that a=a[i] (mod n[i]).

    Args:
        a: List of a[i], the residues modulo n[i]
        n: List of n[i], all n[i] should be pairwise coprime.

    Returns:
        int: The unique value of (a modulo product of all n[i]) such that a=a[i] (mod n[i]).

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


def is_quadratic_residue_prime(a: int, p: int) -> int:
    """Checks whether integer a is a quadratic residue modulo prime p.

    Args:
        a: The integer to be checked.
        p: A prime number.

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
    

def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    """Checks whether integer a is a quadratic residue modulo p^e.

    Args:
        a: The integer to be checked.
        p: A prime number.
        e: The power to which p is to be raised. (e>=1)

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


def floor_sqrt(x: int) -> int:
    """ Returns the floor of the square root of x.

    Args:
        x: The input non-negative integer, whose sqrt is to be found.

    Returns:
        int: The floor of the square root of x.
    
    """
    length: int = len(bin(x)[2:])
    k: int = (length-1)//2
    sqrt: int = 1<<k

    for i in range(k-1, -1, -1):
        if pow((sqrt + (1<<i)), 2) <= x:
            sqrt += 1<<i

    return sqrt

# print(floor_sqrt(123456))


def floor_root(x: int, p: int) -> int:
    """ Returns the floor of the p'th root of x.

    Args:
        x: The non-negative integer, whose root is to be found.
        p: A positive integer, such that floor of x^(1/p) is returned.

    Returns:
        int: Floor of x^(1/p).
    
    """
    length: int = len(bin(x)[2:])
    k: int = (length-1)//p
    root: int = 1<<k

    for i in range(k-1, -1, -1):
        if pow((root + (1<<i)), p) <= x:
            root += 1<<i

    return root

# print(floor_root(1234, 10))

def is_perfect_power(x: int) -> bool:
    """ Checks if x is a perfect power of an integer or not.

    Args:
        x: An integer

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


def is_in_miller_rabin_set(n: int, a: int) -> bool:
    """ Helper function used in is_prime function
    
    Used to implement Miller-Rabin algorithm

    Args:
        n: The prime number
        a: The random number which should be checked if it is in a set or not
            The set is defined in the Miller-Rabin algorithm as 
            L_n:={a in Z_n\{0}: pow(a, t*pow(2, h), n) == 1 and
                pow(a, t*pow(2, j+1))%n = 1 => pow(a, t*pow(2, j))%n == +-1 for j = 0, ..., h-1} 

    Returns:
        bool: True if a is in the set, False otherwise
    
    """
    t: int = n-1
    h: int = 0
    while t%2 == 0:
        t = t//2
        h+=1
    b: int = pow(a, t, n)
    if b == 1: return True
    for j in range(0, h):
        if b == n-1 or b == -1 : return True
        if b == 1: return False
        b = pow(b, 2, n)
    return False


def is_prime(n: int) -> bool:
    """ Checks if n is prime or not.

    Args:
        n: A positive integer

    Returns:
        bool: True if n is prime, False otherwise.
    
    The function uses Miller-Rabin algorithm.
    """
    k: int = 100

    for i in range(0, k):
        a: int = random.randint(1, n-1)
        if is_in_miller_rabin_set(n, a) == False:
            return False
    
    return True

# print(is_prime(8683317618811886495518194401279999999))
# print(pow(1234, 123, 123456))


def gen_prime(m: int) -> int:
    """ Generates a random prime less than or equal to m.

    Args:
        m: The upper bound for the random prime

    Returns:
        int: A random prime in the range [2, m]
    
    This function uses the Niller-Rabin algorithm
    """
    while True:
        n: int = random.randint(2, m)

        is_prime_trial: bool = True
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
        k: A positive integer

    Returns:
        int: A random k-bit prime
    
    This function uses the Miller-Rabin algorithm
    """
    while True:
        n: int = random.randint(pow(2, k-1), pow(2, k)-1)
        
        is_prime_trial: bool = True
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


def find_factor(n: int) -> int:
    """ Helper function for factor function, it finds a factor of n by trial division.

    Args:
        n: The integer whose facor is to be found

    Returns:
        int: The least prime factor of n
    
    """
    for i in range(2, floor_sqrt(n)+1):
        if n%i == 0: return i
    return 0


def factor(n: int) -> list[tuple[int, int]]:
    """ Returns the prime factorisation of n.

    Args:
        n: A positive integer.

    Returns:
        list[tuple[int, int]]: The factorisation of n
            Each tuple in the list is of the form (p, e), where p is a prime which divides n and e is the exponent of p in n.
            The list is sorted in ascending order of the first element of each tuple.
    
    """
    if n == 1: return []

    if is_prime(n):
        return [(n, 1)]
    
    d: int = find_factor(n)
    # print(d, n//d)
    n_factorisation: dict[int, int] = dict(factor(n//d))

    new_factors: dict[int, int] = dict(factor(d))
    for p in new_factors:
        if p in n_factorisation:
            n_factorisation[p] += new_factors[p]
        else:
            n_factorisation[p] = new_factors[p]

    factors: list[tuple[int, int]] = []
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
    phi: int = 1
    factors: list[tuple[int, int]] = factor(n)
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

    A is m*n, each element of A is a row
    m = no. of rows, n = no. of columns
    A[i-1] is ith row, A[:][j-1] is jth column

    The fractions.Fraction class is used.
    """
    B: list[list[fractions.Fraction]] = [[fractions.Fraction(A[i][j], 1) for j in range(len(A[i]))] 
                for i in range(len(A))]
    r: int = 0
    m: int = len(A)
    n: int = len(A[0])
    for j in range(1, n+1):
        l: int = 0
        i: int = r
        while l == 0 and i < m:
            i += 1
            # print(i)
            if B[i-1][j-1] != 0: l = i
        if l != 0:
            r = r+1
            swap = B[r-1][:]
            B[r-1] = B[l-1][:]
            B[l-1] = swap[:]
            b: fractions.Fraction = 1/B[r-1][j-1]
            for iter in range(len(B[r-1])): B[r-1][iter] *= b
            for i in range(1, m+1):
                if i != r:
                    b1: fractions.Fraction = B[i-1][j-1]
                    for iter in range(len(B[i-1])): B[i-1][iter] -= b1*B[r-1][iter]            

    return B


class QuotientPolynomialRing:
    """ This class represents a polynomial quotient ring.

    An element in this polynomial ring is represented by a list of integers p, 
    where p[i] is the "i"th coefficient.

    Attributes:
        pi_generator: This is the modulus of the polynomial ring
        element: This is the element of the quotient polynomial ring of modulus pi_generator
    """

    pi_generator: list[int]

    element: list[int]

    def __init__(self, poly: list[int], pi_gen:list[int]) -> None:
        """ Initializes the instance of polynomial based on the given element and pi_generator

        Args:
            poly: The element of the quotient ring
            pi_gen: The polynomial based on which the ring is formed

        Raises:
            ValueError("Empty pi_generator"):
                If the pi_generator is an empty list
            ValueError("Pi_generator is not monic"): 
                If the pi_generator is not a monic polynomial

        """
        if pi_gen == []:
            raise ValueError('Empty pi_generator')
        if pi_gen[len(pi_gen)-1] != 1:
            raise ValueError('Pi_generator is not monic')
        self.pi_generator = pi_gen
        self.element = poly
        return

    @staticmethod
    def Deg(p: list[int]) -> int:
        """ Returns the degree of a polynomial represented by a list

        Args:
            p: The polynomial

        Returns:
            int: The degree of the polynomial
        
        """
        if len(p) == 0: return 0
        for i in range(len(p)-1, -1, -1):
            if p[i]!= 0: return i
        return 0

    @staticmethod
    def Mod_gen(p: list[int], gen: list[int]) -> "QuotientPolynomialRing":
        """ Returns the representative of the polynomial p in the ring made by gen

        Args:
            p: The polynomial whose representative is to be found
            gen: The polynomial using which the ring is made

        Returns:
            QuotiemtPolynomialRing: The representative of p with pi_generator as gen
        """
        while len(p) >= len(gen):
            order_diff: int = len(p)-len(gen)
            for i in range(len(p)-order_diff):
                p[i+order_diff] -= p[len(p)-1]*gen[i]
            p.pop()
        return QuotientPolynomialRing(p, gen)

    @staticmethod
    def Add(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing") -> "QuotientPolynomialRing":
        """ Returns the sum of two polynomials in the quotient ring

        Args:
            poly1: The first polynomial
            poly2: The secund polynomial

        Returns:
            QuotientPolynomialRing: The sum of the two polynomials

        Raises:
            ValueError: If the two polynomials have different pi_generators
        
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        
        sum: list[int] = []

        l1: int = min(len(poly1.element), len(poly2.element))
        l2: int = max(len(poly1.element), len(poly2.element))
        Max: "QuotientPolynomialRing" = poly1
        if len(poly1.element) == l2: Max = poly1
        else: Max = poly2

        for i in range(l1):
            sum.append(poly1.element[i]+poly2.element[i])

        for i in range(l1, l2):
            sum.append(Max.element[i])

        sum_poly: "QuotientPolynomialRing" = QuotientPolynomialRing.Mod_gen(sum, poly1.pi_generator)
        
        return sum_poly
    
    @staticmethod
    def Sub(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing") -> "QuotientPolynomialRing":
        """ Retuns the difference of two polynomials

        Args:
            poly1: The first polynomial
            poly2: The second polynomial

        Returns:
            QuotientPolynomialRing: The difference of the two polynomials

        Raises:
            ValueError: If the polynomials have different pi_generators
        
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        
        e1: list[int] = poly1.element
        e2: list[int] = poly2.element
        l: int = 0
        longer: list[int] = []
        if len(e1) > len(e2): 
            l = len(e2)
            longer = e1
        else: 
            l = len(e1)
            longer = e2
        diff: list[int] = []
        for i in range(l):
            diff.append(e1[i]-e2[i])

        if longer == e1:
                for i in range(l, len(longer)):
                    diff.append(longer[i])
        else: 
            for i in range(l, len(longer)):
                diff.append(-longer[i])

        diff_poly: "QuotientPolynomialRing" = QuotientPolynomialRing.Mod_gen(diff, poly1.pi_generator)

        return diff_poly
    
    @staticmethod
    def Mul(poly1: "QuotientPolynomialRing", poly2: "QuotientPolynomialRing") -> "QuotientPolynomialRing":
        """ Returns the product of two polynomials in the ring

        Args:
            poly1: The first polynomial
            poly2: The second polynomial

        Returns:
            QuotientPolynomialRing: The product of the two polynomials

        Raises:
            ValueError: If the two polynomials have different pi_generators
        
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        deg: int = 2*(len(poly1.element)-1)
        prod: list[int] = [0 for x in range(deg+1)]
        for i in range(len(poly1.element)):
            for j in range(len(poly2.element)):
                prod[i+j] += poly1.element[i]*poly2.element[j]

        prod_poly: "QuotientPolynomialRing" = QuotientPolynomialRing.Mod_gen(prod, poly1.pi_generator)

        return prod_poly
    
    @staticmethod
    def Scale(p: list[int], a: int) -> list[int]:
        """ Scales each element of a list p by a factor a

        Args:
            p: The list to be scaled
            a: The scaling factor

        Returns:
            list[int]: The scaled list
        
        """
        for i in range(len(p)):
            p[i] *= a
        return p
    
    @staticmethod
    def Common_factor(p: list[int]) -> int:
        """ Returns the gcd of a list of integers

        The elements with value 0 are ignored

        Args:
            p: A list of integers

        Returns:
            int: The gcd of the list of integers, ignoring the 0s
        
        """
        f: int = p[QuotientPolynomialRing.Deg(p)]
        if f == 0: return 0
        for i in range(len(p)):
            if p[i] != 0: f = pair_gcd(f, p[i])
        return f
    
    @staticmethod
    def Reduce(p: list[int]) -> list[int]:
        """ Reruces a list of integers to the form where the gcd of all integers is 1

        Args:
            p: The list of integers to be reduced to the simplest form

        Returns:
            list[int]: The reduced form of the list

        Example: 
            Input: [2, -4, 2] 
            Output: [1, -2, 1]
        
        """
        l: int = len(p)
        gcd: int = 0
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
        """ Returns the gcd of two polynomials.

        Args:
            poly1: The first polynomial
            poly2: The second polynomial

        Returns:
            QuotientPolynomialRing: The gcd of the two polynomials in the polynomial quotient ring

        Raises:
            ValueError: If the two polynomials have different pi_generators
        
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Different pi_generators")
        a: int = QuotientPolynomialRing.Common_factor(poly1.element)
        b: int = QuotientPolynomialRing.Common_factor(poly2.element)
        # print(a, b)
        f: int = pair_gcd(a, b)
        p1: list[int] = poly1.element  # Assuming neither p1 nor y are zero polynomials
        p2: list[int] = poly2.element
        pow_X_1: int = 0
        while p1[0] == 0:
            for i in range(len(p1)-1): p1[i] = p1[i+1]
            p1[len(p1)-1] = 0
            pow_X_1 += 1
        pow_X_2: int = 0
        while p2[0] == 0:
            for i in range(len(p2)-1): p2[i] = p2[i+1]
            p2[len(p2)-1] = 0
            pow_X_2 += 1
        # return min(pow_X_1, pow_X_2)

        x: list[int] = QuotientPolynomialRing.Reduce(p1)
        y: list[int] = QuotientPolynomialRing.Reduce(p2)
        zeroes: list[int] = [0 for i in range(len(x))] # Assuming len(x) == len(y)
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
                swap: list[int] = x; x = y; y = swap
                # print(1)
        
            # print(x, y)
            if y[0] % x[0] == 0:
                c: int = y[0]//x[0]
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
                x = QuotientPolynomialRing.Scale(x, a); y1: list[int] = QuotientPolynomialRing.Scale(y, b)
                for i in range(len(x)): x[i] += y1[i]
                for i in range(len(x)): y[i] -= x[i]*(y[0]//x[0])
                # print(x[0], y[0])
            x = QuotientPolynomialRing.Reduce(x)
            y = QuotientPolynomialRing.Reduce(y)
            # print(x, y)
            # print()
    
        x = QuotientPolynomialRing.Scale(x, f)
        deg = min(pow_X_1, pow_X_2)
        for p in range(deg):
            for i in range(len(x)-1, -1, -1): x[i] = x[i-1] 
            x[0] = 0
        # print(x)
        return QuotientPolynomialRing(x, poly1.pi_generator)
    
    @staticmethod
    def Inv(poly: "QuotientPolynomialRing")->"QuotientPolynomialRing":
        """ Returns the modular inverse of a polynomial in a quotient ring.

        Args:
            poly: The polynomial

        Returns:
            The inverse of the polynomial

        Raises:
            ValueError: If the modular inverse does not exist

        The fractions.Fraction class from fractions module is used.
        
        """
        p: list[int] = poly.element[:]
        base: list[int] = poly.pi_generator[:]
        d1: int =QuotientPolynomialRing.Deg(p)
        d2: int = QuotientPolynomialRing.Deg(base)
        
        # Creating matrix for solving equations
        A: list[list[int]] = []
        for iter1 in range(d1+d2):
            # print(1, iter1)
            Ai: list[int] = [0 for i in range(d1+d2)]
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
        B: list[list[fractions.Fraction]] = gaussian_elimination(A)
        # print(B)

        inv: list[int] = [0 for i in range(d2)]
        null: list[fractions.Fraction] = [fractions.Fraction(0, 1) for i in range(d1+d2+1)]
        for iter in range(d1+d2):
            if B[iter] == null: raise ValueError("Inverse does not exist")
            if B[iter][d1+d2].denominator != 1: raise ValueError("Inverse does not exist")  
        for iter in range(d2):
            inv[iter] = int(B[iter][d1+d2])
        # print(inv)
        return QuotientPolynomialRing(inv, base)

# print(QuotientPolynomialRing.Inv(QuotientPolynomialRing([1, 2, 1], [0, 3, 3, 1])).element)

# x = QuotientPolynomialRing([1, 3, 3, 1], [0, 1, 1])
# y = QuotientPolynomialRing([2, 4, 2], [0, 1, 1])

# print(QuotientPolynomialRing.Mod([3, 7, 5, 1], [0, 1, 1]))
# print(QuotientPolynomialRing.Sub(x, y).element)
# print(QuotientPolynomialRing.Add(x, y).element, QuotientPolynomialRing.Add(x, y).pi_generator)
# print(QuotientPolynomialRing.GCD(x, y).element)
# print(QuotientPolynomialRing.Inv(QuotientPolynomialRing([0, 0, 1, 0], [7, 0, 0, 3, 1])).element)

# p1 = QuotientPolynomialRing([-3, -5, -1, 1], [7, 0, 0,  3, 1])
# p2 = QuotientPolynomialRing([1, 5, 7, 3], [7, 0, 0,  3, 1])
# print(QuotientPolynomialRing.GCD(p1, p2).element)
# print(QuotientPolynomialRing.Mod([-3, -5, -1, 1], [1, 5, 7, 3], 4))


def mod_base(p: list[int], r: int) -> list[int]:
    """ Helper function for aks_test
    
    """
    while len(p) >= r+1:
        order_diff: int = len(p)-(r+1)
        p[len(p)-r-1] += p[len(p)-1]
        p.pop()

    return p

def mul(p1: list[int], p2: list[int], r: int, n: int)->list[int]:
    """ Helper function for aks_test
    
    """
    # print("a")
    l1: int = len(p1); l2: int = len(p2)
    # print((l1-1)+(l2-1)+1)
    prod: list[int] = [0 for i in range((l1-1)+(l2-1)+1)]
    for i in range(l1):
        for j in range(l2):
            prod[i+j] += (p1[i]*p2[j])%n
            prod[i+j] %= n
    prod = mod_base(prod, r)
    return prod

def fast_exp(p,power,r, n):
    """ Helper function for aks_test
    
    """
    # print(power)
    if power == 0: return [1]
    elif power%2 == 0:
        return fast_exp(mul(p, p, r, n), power//2, r, n)
    else:
        return mul(fast_exp(mul(p, p, r, n), (power-1)//2, r, n), p, r, n)


def aks_satisfied(j: int, r: int, n: int) -> bool:
    """ Helper function for aks_test
    """
    # print(1)
    x: list[int] = fast_exp([j, 1], n, r, n)
    # print(2)
    y: list[int] = fast_exp([0, 1], n, r, n)
    
    l1: int = len(x)
    l2: int = len(y)

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
    """ Checks whether a number is prime or not.

    Uses the deterministic, polynomial-time, AKS primality test.
    
    Args:
        n: The integer to be tested

    Returns:
        bool: True if the number is a prime, False otherwise

    If n is prime the algorithm may take an extremely long time    
    """
    if is_perfect_power(n): return False

    r: int = 2
    while True:
        if pair_gcd(n, r) != 1: break
        else:
            o: int = 1
            pow_n: int = n
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
    q: int = p-1
    m: int = floor_sqrt(q)
    values: dict[int, int] = {}
    value: int = 1
    for i in range(m+1):
        values[value] = i
        value *= g
        value %= p
    # print(values)
    value = x
    mul: int = pow(g, q-m, p)
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
        a: An integer
        p: A prime number

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
        a: An integer
        n: a positive integer

    Returns:
        int: The Jacobi symbol (a | n)
    
    """
    if pair_gcd(a, n) != 1: return 0
    j: int = 1
    factors = factor(n)
    # print(factors)
    for q, i in factors:
        # print(legendre_symbol(a, q))
        j *= pow(legendre_symbol(a, q), i)
    return j


def modular_sqrt_prime(x: int, p: int)->int:
    """ Returns the modular square root of a number modulo a prime number

    Args:
        x: The number 
        p: The prime number, where square root is found in Z_p

    Returns:
        int: Modular square root of x modulo p

    Raises:
        ValueError: Raised if the modular square root does not exist
    """
    if is_quadratic_residue_prime(x, p) != 1:
        raise ValueError("Modular square root does not exist")
    if p == 2: return 1
    if p%4 == 3:
        p1: int = pow(x, (p+1)//4, p)
        if p1 > p-p1: return p-p1
        else: return p1

    c: int = 2
    while is_quadratic_residue_prime(c,p) != -1: c = random.randint(2, p-1)

    h: int = 0
    m: int = p-1
    while m%2 == 0:
        m = m//2
        h += 1
    c = pow(c, m, p)
    x1: int = pow(x, m, p)
    a: int = discrete_log(x1, c, p)
    
    p1 = (pow(c, a//2, p)*pow(x, -(m//2), p))%p
    if p1 < p-p1: return p1
    else: return p-p1


def modular_sqrt_prime_power(x: int, p: int, e: int)->int: # assuming odd prime
    """ Returns the modular square root of x modulo p^e

    The least modular square root is returned

    Args:
        x: The number whose modular square root is to be found
        p: The prime number
        e: The power to which p is to be raised to get the modulus

    Returns:
        int: The least square root of x in the ring Z_(p^e)

    Raises:
        ValueError: Raised if the modular square root does not exist
    
    """
    if x%pow(p, e) == 0: return 0
    if is_quadratic_residue_prime_power(x, p, e) != 1:
        raise ValueError("Modular square root does not exist")
    
    b: int = modular_sqrt_prime(x, p)
    for i in range(1, e):
        a1: int = mod_inv(2*b, p)
        b1: int = (x-b*b)//pow(p, i)
        h: int = (a1*b1)%p
        b = (b + h*pow(p, i))%pow(p, i+1)
    p1: int = b
    if p1 > pow(p, e)-p1: return pow(p, e)-p1
    else: return p1


def modular_sqrt_2_pow(x: int, e: int)->list[int]:
    """ Returns a list of all modular square roots of x modulo 2^e

    Args:
        x: The number whose square root is to be found
        e: The power to which 2 is to be raised to get the modulus

    Returns:
        list[int]: A list of all the square roots in the ring Z_(2^e)

    Raises:
        ValueError: Raised if the modular square root does not exist
    
    """
    sqrts: list[int] = []
    num: int = pow(2, e)
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


def modular_sqrt(x: int, z: int) -> int:
    """ Returns the modular square root of x modulo z

    Args:
        x: The number whose square root is to be found
        z: The modulus of the ring in which modular square roots are to be found

    Returns:
        int: The least square root of x in the ring Z_z

    Raises:
        ValueError: Raised if the modular square root does not exist
    
    """
    y: int = z
    pow_2: int = 0
    while y%2 == 0:
        pow_2+=1
        y = y//2

    factors: list[tuple[int, int]] = factor(y)
    sq_roots: list[int] = []

    for prime, power in factors:
        try:
            sq_roots.append(modular_sqrt_prime_power(x%pow(prime, power), prime, power))
        except:
            raise ValueError("Modular square root does not exist")

    if pow_2 != 0:
        factors.append((2, pow_2))
        try:
            sqrts_2: list[int] = modular_sqrt_2_pow(x, pow_2)
        except:
            raise ValueError("Modular square root does not exist")
        # print(sqrts_2)

    n: list[int] = [pow(prime, power) for prime, power in factors]
    
    # For listing all possibilities of sq. roots modulo prime-powers
    if len(sq_roots) != 0:
        a_possible: list[list[int]] = [[sq_roots[0]], [-sq_roots[0]]] 
        for i in range(1, len(sq_roots)):
            t: int = len(a_possible)
            for j in range(t):
                a1: list[int] = a_possible[j][:]
                a_possible[j].append(sq_roots[i])
                a1.append(-sq_roots[i])
                a_possible.append(a1)

        if pow_2 != 0:
            t = len(a_possible)
            for i in range(t):
                a1 = a_possible[i][:]
                a_possible[i].append(sqrts_2[0])
                for j in range(1, len(sqrts_2)):
                    a2: list[int] = a1[:]
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


def is_smooth(m: int, y: int) -> bool:
    """ Checks whether m is y-smooth or not

    Args:
        m: the number to be checked
        y: the number with respect to which smoothness is to be checked

    Returns:
        bool: True if m is y-smooth, False otherwise
    
    """
    for i in range(2, y+1)  :
        if m%i == 0:
            while m%i == 0:
                m = m//i
        if m <= y: return True
    return False


def gaussian_elimination_modulo_p(A: list[list[int]], p: int) -> list[list[int]]:
    """ Performs gaussian elimination in the ring Z_p, where p is a prime number

    Used to implement the linear algebra in the function prob_find_factor

    Args:
        A: the matrix on which gaussian elimination is to be performed
        p: the prime which forms the ring Z_p

    Returns:
        list[list[int]]: The matrix after the operation


    A is m*n, each element of A is a row
    m = no. of rows, n = no. of columns
    A[i-1] is ith row, A[:][j-1] is jth column
    """
    B: list[list[int]] = [[A[i][j] for j in range(len(A[i]))] 
                for i in range(len(A))]
    r: int = 0
    m: int = len(A)
    n: int = len(A[0])
    for j in range(1, n+1):
        l: int = 0
        i: int = r
        while l == 0 and i < m:
            i += 1
            # print(i)
            if B[i-1][j-1]%p != 0: l = i
        if l != 0:
            r = r+1
            swap: list[int] = B[r-1][:]
            B[r-1] = B[l-1][:]
            B[l-1] = swap[:]
            b: int = mod_inv(B[r-1][j-1], p)
            for iter in range(len(B[r-1])): 
                B[r-1][iter] *= b
                B[r-1][iter] %= p
            for i in range(1, m+1):
                if i != r:
                    b1: int = B[i-1][j-1]
                    for iter in range(len(B[i-1])): 
                        B[i-1][iter] -= b1*B[r-1][iter]
                        B[i-1][iter] %= p            

    return B

def prob_find_factor(n: int)->int:
    """ Returns a random non-trivial (i.e., not 1 or n) factor of the number n

    Args:
        n: The number whose factor is to be found

    Returns:
        int: A non-trivial factor of the number

    A subexponential factoring algorithm is implemented in this function
    
    """
    y: int = pow(2, floor_sqrt((len(bin(n)[2:])*len(bin(len(bin(n)[2:]))[2:]))//2))
    primes: list[int] = []
    for i in range(2, y+1):
        if is_prime(i): primes.append(i)
    k: int = len(primes)
    # print(y, primes, k)

    v: list[list[int]] = []
    a: list[int] = []

    d: int = random.randint(1, n-1)
    while pair_gcd(d, n) != 1: d = random.randint(1, n-1)
    # print(d)

    counter_lim: int = 10000 # I am getting lucky with this limit, I need it to prevent an infinite loop

    i = 0
    while True:
        i = i+1
        # print(i)
        counter: int = 0
        while counter < counter_lim:
            counter+=1
            a_i: int = random.randint(1, n-1)
            while pair_gcd(a_i, n) != 1: a_i = random.randint(1, n-1)
            m_i: int = (a_i*a_i*d)%n
            # print(a_i, m_i)
            e_i: list[int] = [0 for iter in range(k)]
            for iter in range(k):
                pow_p_i: int = 0
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

    A: list[list[int]] = [[v[i][j] for i in range(k+2)] for j in range(k+1)]
    G: list[list[int]] = gaussian_elimination_modulo_p(A, 2)

    # print(G)

    zeroes: list[int] = [0 for i in range(len(A[0]))]
    not_possible: list[int] = zeroes[:]; not_possible[k] = 1

    c: list[int] = []
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

    alpha: int = 1

    for i in range(k+2):
        alpha *= pow(a[i], c[i])
    # print(alpha, d)

    e: list[int] = [0 for i in range(k+1)]
    for i in range(k+1):
        for j in range(k+2):
            e[i] += c[j]*v[j][i]

    # print(e)

    beta: int = 1
    for i in range(k):
        beta *= pow(primes[i], e[i]//2)
    # beta //= pow(d, e[k]//2)

    # print(alpha, beta)

    gamma: int = (alpha//beta)*pow(d, e[k]//2)

    # print(gamma)
    if gamma%n == 1 or gamma%n == -1:
        # print("noooooooooooooooo4")
        return prob_find_factor(n)
    
    factor: int = pair_gcd((gamma-1)%n, n)
    if factor == 1: 
        # print("noooooooooooooooooooo5")
        return prob_find_factor(n)
    return factor

# print(prob_find_factor(1408198281))

def probabilistic_factor(n: int) -> list[tuple[int, int]]:
    """ Returns the prime factorisation of n.

    Args:
        n: A positive integer.

    Returns:
        list[tuple[int, int]]: The prime factorisation of n
            Each tuple in the list is of the form (p, e),
                where p is a prime which divides n and e is the exponent of p in n.
            The list is sorted in ascending order of the first element of each tuple.
    
    """
    factors: list[tuple[int, int]] = []
    # to ensure that n is odd
    pow_2: int = 0
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
    length: int = len(bin(n)[2:])
    pow_range: int = length-1
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
    y: int = pow(2, floor_sqrt((len(bin(n)[2:])*len(bin(len(bin(n)[2:]))[2:]))//2))
    for i in range(2, y+1):
        pow_i: int = 0
        while n%i == 0:
            pow_i += 1
            n = n//i
        if pow_i != 0: factors.append((i, pow_i))
    if n == 1: return factors

    # The actual algorithm

    if is_prime(n):
        factors.append((n, 1))
        return factors
    
    d: int = prob_find_factor(n)
    # print(d, n//d)
    n_factorization: dict[int, int] = dict(probabilistic_factor(n//d))
    new_factors: dict[int, int] = dict(probabilistic_factor(d))
    for prime in new_factors:
        if prime in n_factorization:
            n_factorization[prime] += new_factors[prime]
        else:
            n_factorization[prime] = new_factors[prime]

    dict_factors: dict[int, int] = dict(factors)
    for prime in n_factorization:
        if prime in dict_factors:
            dict_factors[prime] += n_factorization[prime]
        else:
            dict_factors[prime] = n_factorization[prime]

    factors_final: list[tuple[int, int]] = []
    for prime in sorted(dict_factors):
        factors_final.append((prime, dict_factors[prime]))

    return factors_final
