# This function gives our structure of skeletal model


def getSkeletalModelStructure():
    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone)
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.
    #
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return (
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        # (3, 4, 3),

        # right shoulder
        (1, 4, 1),

        # right arm
        (4, 5, 2),
        # (6, 7, 3),

        # left hand - wrist
        (5, 6, 3),

        # left hand - palm
        (6, 7, 4),
        (6, 11, 8),
        (6, 15, 12),
        (6, 19, 16),
        (6, 23, 20),

        # left hand - 1st finger
        (7, 8, 5),
        (8, 9, 6),
        (9, 10, 7),

        # left hand - 2nd finger
        (11, 12, 9),
        (12, 13, 10),
        (13, 14, 11),

        # left hand - 3rd finger
        (15, 16, 13),
        (16, 17, 14),
        (17, 18, 15),

        # left hand - 4th finger
        (19, 20, 17),
        (20, 21, 18),
        (21, 22, 19),

        # left hand - 5th finger
        (23, 24, 21),
        (24, 25, 22),
        (25, 26, 23),

        # right hand - wrist
        (3, 27, 3),

        # right hand - palm
        (27, 28, 4),
        (27, 32, 8),
        (27, 36, 12),
        (27, 40, 16),
        (27, 44, 20),

        # right hand - 1st finger
        (28, 29, 5),
        (29, 30, 6),
        (30, 31, 7),

        # right hand - 2nd finger
        (32, 33, 9),
        (33, 34, 10),
        (34, 35, 11),

        # right hand - 3rd finger
        (36, 37, 13),
        (37, 38, 14),
        (38, 39, 15),

        # right hand - 4th finger
        (40, 41, 17),
        (41, 42, 18),
        (42, 43, 19),

        # right hand - 5th finger
        (44, 45, 21),
        (45, 46, 22),
        (46, 47, 23),
    )


def getMTCSkeletalModelStructure():
    return (
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        (3, 27, 3),

        # right shoulder
        (1, 4, 1),

        # right arm
        (4, 5, 2),
        (5, 6, 3),

        # left hand - palm
        (6, 7, 5),
        (6, 11, 9),
        (6, 15, 13),
        (6, 19, 17),
        (6, 23, 21),

        # left hand - 1st finger
        (7, 8, 6),
        (8, 9, 7),
        (9, 10, 8),

        # left hand - 2nd finger
        (11, 12, 10),
        (12, 13, 11),
        (13, 14, 12),

        # left hand - 3rd finger
        (15, 16, 14),
        (16, 17, 15),
        (17, 18, 16),

        # left hand - 4th finger
        (19, 20, 18),
        (20, 21, 19),
        (21, 22, 20),

        # left hand - 5th finger
        (23, 24, 22),
        (24, 25, 23),
        (25, 26, 24),

        # right hand - palm
        (27, 28, 5),
        (27, 32, 9),
        (27, 36, 13),
        (27, 40, 17),
        (27, 44, 21),

        # right hand - 1st finger
        (28, 29, 6),
        (29, 30, 7),
        (30, 31, 8),

        # right hand - 2nd finger
        (32, 33, 10),
        (33, 34, 11),
        (34, 35, 12),

        # right hand - 3rd finger
        (36, 37, 14),
        (37, 38, 15),
        (38, 39, 16),

        # right hand - 4th finger
        (40, 41, 18),
        (41, 42, 19),
        (42, 43, 20),

        # right hand - 5th finger
        (44, 45, 22),
        (45, 46, 23),
        (46, 47, 24),
    )


# Computing number of joints and limbs
def structureStats(structure):
    ps = {}
    ls = {}
    for a, b, l in structure:
        ps[a] = "gotcha"
        ps[b] = "gotcha"
        ls[l] = "gotcha"
    return len(ls), len(ps)
