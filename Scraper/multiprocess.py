from multiprocessing import Pool, cpu_count


def test(x):
    return x * 2

if __name__ == '__main__':
    a = [[i for i in range(5)] for a in range(10)]

    p = Pool(processes=cpu_count())

    b = p.map(test, a)

    print(b)
