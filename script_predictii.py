import cv2 as cv
import numpy as np
import os


def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def galben(i, j):
    if (i == 0 and j == 0) or (i == 6 and j == 0) or (i == 7 and j == 0) or (i == 13 and j == 0) or (
            i == 13 and j == 6) or (i == 13 and j == 7) or (i == 13 and j == 13) or (i == 7 and j == 13) or (
            i == 6 and j == 13) or (i == 0 and j == 13) or (i == 0 and j == 7) or (i == 0 and j == 6):
        return True
    return False


def tabla_start():
    tabla = [
        ['x3', '', '', '', '', '', 'x3', 'x3', '', '', '', '', '', 'x3'],
        ['', 'x2', '', '', '/', '', '', '', '', '/', '', '', 'x2', ''],
        ['', '', 'x2', '', '', '-', '', '', '-', '', '', 'x2', '', ''],
        ['', '', '', 'x2', '', '', '+', '*', '', '', 'x2', '', '', ''],
        ['', '/', '', '', 'x2', '', '*', '+', '', 'x2', '', '', '/', ''],
        ['', '', '-', '', '', '', '', '', '', '', '', '-', '', ''],
        ['x3', '', '', '*', '+', '', 1, 2, '', '*', '+', '', '', 'x3'],
        ['x3', '', '', '+', '*', '', 3, 4, '', '+', '*', '', '', 'x3'],
        ['', '', '-', '', '', '', '', '', '', '', '', '-', '', ''],
        ['', '/', '', '', 'x2', '', '+', '*', '', 'x2', '', '', '/', ''],
        ['', '', '', 'x2', '', '', '*', '+', '', '', 'x2', '', '', ''],
        ['', '', 'x2', '', '', '-', '', '', '-', '', '', 'x2', '', ''],
        ['', 'x2', '', '', '/', '', '', '', '', '/', '', '', 'x2', ''],
        ['x3', '', '', '', '', '', 'x3', 'x3', '', '', '', '', '', 'x3']
    ]

    return tabla


def extrage_careu(image):
    if image is None:
        print("Error: Image not loaded.")
        return None
    blue_channel, green_channel, red_channel = cv.split(image)
    gray = blue_channel
    blurred = cv.GaussianBlur(gray, (5, 5), 7)
    _, thresh = cv.threshold(blurred, 150, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(max_contour)
    cropped_board = image[y + 250:y + h - 250, x + 250:x + w - 250]
    # show_image('cropped_board',cropped_board)
    return cropped_board


def extrage_patrat(img1, a, b):
    width1, height1 = img1.shape[:2]
    i, j = a, b
    piece1 = img1[0 + i * (height1 // 14) + 10: (i + 1) * (height1 // 14) - 10,
             0 + j * (width1 // 14) + 10: (j + 1) * (width1 // 14) - 10]
    piece1 = cv.GaussianBlur(piece1, (3, 3), 3)
    piece1 = cv.cvtColor(piece1, cv.COLOR_BGR2GRAY)
    # piece1 = cv.erode(piece1, np.ones((3, 3), np.uint8))
    # blue_channel, green_channel, red_channel = cv.split(piece1)
    _, piece1 = cv.threshold(piece1, 100, 255, cv.THRESH_BINARY)
    # show_image("template_img", piece1)
    return piece1


def extrage_piese(img1, img2):
    cropped_board1 = extrage_careu(img1)
    cropped_board2 = extrage_careu(img2)
    width1, height1 = cropped_board1.shape[:2]
    width2, height2 = cropped_board2.shape[:2]

    medie = []
    for i in range(0, 14):
        for j in range(0, 14):
            piece1 = cropped_board1[0 + i * (height1 // 14) + 10: (i + 1) * (height1 // 14) - 10,
                     0 + j * (width1 // 14) + 10: (j + 1) * (width1 // 14) - 10]
            piece2 = cropped_board2[0 + i * (height1 // 14) + 10: (i + 1) * (height1 // 14) - 10,
                     0 + j * (width1 // 14) + 10: (j + 1) * (width1 // 14) - 10]
            if galben(i, j):
                piece1 = cv.cvtColor(piece1, cv.COLOR_BGR2GRAY)
                piece2 = cv.cvtColor(piece2, cv.COLOR_BGR2GRAY)
                _, thresh1 = cv.threshold(piece1, 100, 255, cv.THRESH_BINARY)
                _, thresh2 = cv.threshold(piece2, 100, 255, cv.THRESH_BINARY)
                # show_image("Galben", thresh1)
            else:
                blue_channel, green_channel, red_channel = cv.split(piece1)
                _, thresh1 = cv.threshold(blue_channel, 100, 255, cv.THRESH_BINARY)
                blue_channel, green_channel, red_channel = cv.split(piece2)
                _, thresh2 = cv.threshold(blue_channel, 100, 255, cv.THRESH_BINARY)
            Medie_patch1 = np.mean(thresh1)
            Medie_patch2 = np.mean(thresh2)
            medie.append((Medie_patch2 - Medie_patch1, i + 1, chr(j + 65)))

    minn = 0
    for val in medie:
        if val[0] < minn:
            minn = val[0]

    for val in medie:
        if val[0] == minn:
            sol = identify_piece(extrage_patrat(extrage_careu(img2), val[1] - 1, ord(val[2]) - 65))
            return (val[1], val[2]), int(sol[1])


def identify_piece(target_img):
    maxi = -np.inf
    sol2 = []
    ok = 1
    fixed_size = (300, 300)
    directory_path = 'templates/'
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    for file in files:
        template_img = cv.imread(f"templates/{file}")
        template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
        _, template_img = cv.threshold(template_img, 150, 255, cv.THRESH_BINARY)

        target_img_margin = cv.resize(target_img, fixed_size, interpolation=cv.INTER_AREA)
        target_img_margin = cv.copyMakeBorder(target_img_margin, 600, 600, 600, 600, cv.BORDER_CONSTANT,
                                              value=(255, 255, 255))
        template_edges = cv.resize(template_img, fixed_size, interpolation=cv.INTER_AREA)

        corr = cv.matchTemplate(target_img_margin, template_edges, cv.TM_CCOEFF_NORMED)
        corr = np.max(corr)

        # if ok == 1:
        #     show_image("target_img", target_img_margin)
        #     show_image("template_img", template_edges)
        # if corr>maxi:
        # maxi=corr
        # poz=file
        ok = ok + 1
        file2 = file.strip('.jpg')
        sol2.append((corr, file2))
    return sorted(sol2)[-1]


def interior(number):
    if 0 <= number <= 13:
        return True


def get_score(numar, locatie, tabla):
    lees = [[(0, 1), (0, 2)], [(1, 0), (2, 0)], [(0, -1), (0, -2)], [(-1, 0), (-2, 0)]]
    ecuatie = []
    bonuses = ['x2', 'x3']
    bonus = 1
    operatii = ['*', '+', '/', '-']
    i, j = locatie
    i = i - 1
    j = ord(j) - 65
    pozitie_tabla = tabla[i][j]

    if pozitie_tabla in bonuses:
        bonus = int(pozitie_tabla[1])

    for lee in lees:
        i1 = i + lee[0][0]
        j1 = j + lee[0][1]
        i2 = i + lee[1][0]
        j2 = j + lee[1][1]

        if interior(i1) and interior(j1) and interior(i2) and interior(j2):
            numar1 = tabla[i1][j1]
            numar2 = tabla[i2][j2]
            if type(numar1) == int and type(numar2) == int:
                if numar1 * numar2 == numar:
                    ecuatie.append((numar, "*"))
                elif numar1 + numar2 == numar:
                    ecuatie.append((numar, "+"))
                elif abs(numar1 - numar2) == numar:
                    ecuatie.append((numar, "-"))
                elif numar1 * numar2 != 0:
                    if numar1 // numar2 == numar or numar2 // numar1 == numar:
                        ecuatie.append((numar, "/"))

    if pozitie_tabla in operatii:
        ecuatie = [x for x in ecuatie if x[1] == pozitie_tabla]

    tabla[i][j] = numar
    return sum(x[0] for x in ecuatie) * bonus, tabla


directory_path = 'antrenare/'
numbers = ["0" + str(x) for x in range(1, 10)] + [str(x) for x in range(10, 51)]


def rezolvare(id):
    scor_curent, turn = 0, 0
    tabla = tabla_start()

    turns = [(line.split()[0], int(line.split()[1])) for line in open(f'{directory_path}/{id}_turns.txt')]
    jucator = turns[0][0]

    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    scor_file = open(f"output/{id}_scores.txt", "w")

    for file in files:  # fiecare mutare practic

        file_precedent = ""
        if file.endswith('.jpg') and file[0] == f"{id}":

            file_precedent = f"{id}_{int(file[2:4]) - 1}.jpg"
            if len(file_precedent) == 7:
                file_precedent = f"{file_precedent[:2]}0{file_precedent[2:]}"
            if file.endswith('01.jpg'):
                file_precedent = f"01.jpg"

            # print(file, file_precedent)
            target_img = cv.imread(f"antrenare/{file}")
            template_img = cv.imread(f"antrenare/{file_precedent}")

            locatie, numar = extrage_piese(template_img, target_img)
            print(locatie, numar)
            fisier_mutare = open(f"output/{id}_{numbers[int(file[2:4]) - 1]}.txt", "w")
            fisier_mutare.write(f"{locatie[0]}{locatie[1]} {numar}")

            scor_mutare, tabla = get_score(numar, locatie, tabla)
            # print(scor_mutare)
            scor_curent = scor_curent + scor_mutare

            if turn + 1 < len(turns):
                if int(file[2:4]) + 1 == turns[turn + 1][1]:
                    scor_file.write(f"{jucator} {turns[turn][1]} {scor_curent}\n")
                    jucator = turns[turn + 1][0]
                    scor_curent = 0
                    turn = turn + 1
            elif int(file[2:4]) == 50:
                scor_file.write(f"{jucator} {turns[turn][1]} {scor_curent}")


for i in range(1, 5):
    rezolvare(i)