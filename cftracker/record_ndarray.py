import numpy as np
import cv2


def write_ndarray(filename, array):
    # write down 3-D ndarray into txt file.
    chan, rows, cols = array.shape
    with open(filename, 'w') as f:
        for h in range(chan):
            data = array[h, :, :]
            for i in range(rows):
                for j in range(cols):
                    f.write("%e" % data[i, j])
                    if j < cols - 1:
                        f.write(",")
                f.write("\n")
            if h < chan - 1:
                f.write("\n")


def load_ndarray(filename):
    # read data from txt file and return ndarray.
    res = []
    with open(filename) as f:
        fcontent = f.read()
    fcontent = fcontent.split("\n")
    res_tmp = []
    for bar in fcontent:
        if len(bar) != 0:
            bar = bar.split(",")
            res_c = [float(x) for x in bar]
            res_tmp.append(res_c)
        else:
            res.append(res_tmp)
            res_tmp = []

    return np.array(res)


if __name__ == "__main__":
    # a = [[[1e-5, 2, 3], [4, 5, 6], [7, 8, 9]],[[10, 11, 12], [13, 14, 15], [16, 17, 18]],[[19, 20, 21],
    #                                                                                    [22, 23, 24], [25, 26, 27]]]
    # a = np.array(a)
    # write_ndarray("/home/shawn/scripts_output_tmp/zzz_a.txt", a)
    #
    # b = load_ndarray("/home/shawn/scripts_output_tmp/zzz_a.txt")
    #
    # pass

    # h01 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_H_ifft01.txt")
    # h02 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_H_ifft02.txt")
    # dh = h01 - h02
    #
    # f01 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_f01.txt")
    # f02 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_f02.txt")
    # df = f01 - f02
    #
    # a = np.sum(dh)
    # b = np.sum(df)
    #
    # print("Updated method H and f:")
    # print(a)
    # print(b)
    #
    #
    # h01 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_H01.txt")
    # h02 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_H02.txt")
    # dh = h01-h02

    f01 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_Hlinear01.txt")
    # f02 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_Hlinear02.txt")
    # df = f01-f02

    g01 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_Hlog01.txt")
    # g02 = load_ndarray("/home/shawn/scripts_output_tmp/zzz_Hlog02.txt")
    # dg = g01 - g02

    m_factor = 14.023477597962808
    maxR = 35.35533905932738
    center_pos = (25, 25)

    df, dg = 0, 0

    f01_inversed_last_time = cv2.linearPolar(f01, center_pos, maxR, cv2.WARP_INVERSE_MAP)
    g01_inversed_last_time = cv2.logPolar(g01, center_pos, m_factor, cv2.WARP_INVERSE_MAP)

    count = 0
    while df <= 1e1 or dg <= 1e1:
        f01_inversed = cv2.linearPolar(f01, center_pos, maxR, cv2.WARP_INVERSE_MAP)
        g01_inversed = cv2.logPolar(g01, center_pos, m_factor, cv2.WARP_INVERSE_MAP)
        df = np.sum(f01_inversed - f01_inversed_last_time)
        dg = np.sum(g01_inversed - g01_inversed_last_time)
        f01_inversed_last_time = f01_inversed
        g01_inversed_last_time = g01_inversed
        count += 1
        print(count)
        print(df)
        print(dg)

    # a = np.sum(dh)
    # b = np.sum(df)
    # c = np.sum(dg)
    #
    # print("Init method H H-linear and H-log:")
    # print(a)
    # print(b)
    # print(c)
    #
    # h01 = load_ndarray("/home/shawn/scripts_output_tmp/H_inversed_linear01.txt")
    # h02 = load_ndarray("/home/shawn/scripts_output_tmp/H_inversed_linear02.txt")
    # dh = h01 - h02
    #
    # f01 = load_ndarray("/home/shawn/scripts_output_tmp/H_inversed_log01.txt")
    # f02 = load_ndarray("/home/shawn/scripts_output_tmp/H_inversed_log02.txt")
    # df = f01 - f02
    #
    # a = np.sum(dh)
    # b = np.sum(df)
    #
    # print("Init Inversed method H and f:")
    # print(a)
    # print(b)

    pass