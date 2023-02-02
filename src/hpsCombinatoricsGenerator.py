import copy


class CombinatoricsGenerator:
    def __init__(self, verbosity):
        self.verbosity = verbosity

    def factorial(self, k):
        assert k >= 0
        if k <= 1:
            return 1
        else:
            return k * self.factorial(k - 1)

    def generate(self, k, n):
        if self.verbosity >= 3:
            print("<CombinatoricsGenerator::generate>:")
            print(" k=%i & n=%i" % (k, n))

        if k <= 0 or k > n:
            if self.verbosity >= 3:
                print("combinations = []")
                print("#combinations = 0 (expected = 0)")
            return []

        retVal = []

        digits = [idx for idx in range(k)]

        current_digit = k - 1
        iteration = 0
        while True:
            assert len("".join("%i" % digits[idx] for idx in range(k))) <= len("%i" % n) * k
            retVal.append(copy.deepcopy(digits))
            if digits[current_digit] < (n - (k - current_digit)):
                digits[current_digit] = digits[current_digit] + 1
            else:
                while current_digit >= 0 and digits[current_digit] >= (n - (k - current_digit)):
                    current_digit -= 1
                if current_digit >= 0:
                    digits[current_digit] = digits[current_digit] + 1
                    for idx in range(current_digit + 1, k):
                        digits[idx] = digits[current_digit] + (idx - current_digit)
                    current_digit = k - 1
                else:
                    break
            iteration += 1

        if self.verbosity >= 3:
            print("combinations = %s" % retVal)
            print(
                "#combinations = %i (expected = %i)"
                % (len(retVal), self.factorial(n) / (self.factorial(k) * self.factorial(n - k)))
            )

        return retVal
