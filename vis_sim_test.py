#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # init simulator
    sim = UAV_simulator(np.array([[-10], [0], [100]]))

    # # example of cycling through UAV orientations
    # for i in range(0, 20, 1):
    #     sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    #     sim.UpdatePlots()
    # for j in range(0, 20, 1):
    #     sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    #     sim.UpdatePlots()
    # for k in range(0, 20, 1):
    #     sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
    #     sim.UpdatePlots()

    # # example of cycling through gimbal orientations
    # for i in range(0, 30, 1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    #     sim.UpdatePlots()
    # for j in range(0, -10, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    #     sim.UpdatePlots()
    # for k in range(0, -10, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
    #     sim.UpdatePlots()

    # # example of cycling through gimbal orientations
    # for i in range(0, -20, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    #     sim.UpdatePlots()
    # for j in range(0, -20, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    #     sim.UpdatePlots()
    # for k in range(0, -20, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
    #     sim.UpdatePlots()

    v_r = sim.R_gc @ sim.R_bg @ sim.e1
    v_p = sim.R_gc @ sim.R_bg @ sim.e2
    v_y = sim.R_gc @ sim.R_bg @ sim.e3
    v_yg = sim.R_gc @ sim.e3
    v_pg = sim.R_gc @ sim.e2
    v_rg = sim.R_gc @ sim.e1
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_r)  # right, left, top, bottom
    test = 1

    # test critical roll
    sim.UpdatePertRPY(np.array((ang1, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_r)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((ang2, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_r)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((ang3, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_r)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((ang4, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_r)  # right, left, top, bottom
    test = 1

    sim.UpdatePertRPY(np.array((0, 0, 0)))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_p)  # right, left, top, bottom
    test = 1

    # test critical pitch
    sim.UpdatePertRPY(np.array((0, ang1, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_p)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((0, ang2, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_p)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((0, -ang3, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_p)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((0, -ang4, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_p)  # right, left, top, bottom
    test = 1

    sim.UpdatePertRPY(np.array((0, 0, 0)))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_y)  # right, left, top, bottom
    test = 1

    # test critical yaw
    sim.UpdatePertRPY(np.array((0, 0, -ang1)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_y)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((0, 0, -ang2)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_y)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((0, 0, ang3)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_y)  # right, left, top, bottom
    test = 1
    sim.UpdatePertRPY(np.array((0, 0, ang4)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_y)  # right, left, top, bottom
    test = 1

    sim.UpdatePertRPY(np.array((0, 0, 0)))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_yg)  # right, left, top, bottom
    test = 1

    # test critical gimbal yaw
    sim.UpdateGimbalPertYPR(np.array((-ang1, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_yg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((-ang2, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_yg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((ang3, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_yg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((ang4, 0, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_yg)  # right, left, top, bottom
    test = 1

    sim.UpdateGimbalPertYPR(np.array((0, 0, 0)))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_pg)  # right, left, top, bottom
    test = 1

    # test critical gimbal pitch
    sim.UpdateGimbalPertYPR(np.array((0, ang1, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_pg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((0, ang2, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_pg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((0, -ang3, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_pg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((0, -ang4, 0)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_pg)  # right, left, top, bottom
    test = 1

    sim.UpdateGimbalPertYPR(np.array((0, 0, 0)))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_rg)  # right, left, top, bottom
    test = 1

    # test critical gimbal roll
    sim.UpdateGimbalPertYPR(np.array((0, 0, ang1)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_rg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((0, 0, ang2)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_rg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((0, 0, ang3)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_rg)  # right, left, top, bottom
    test = 1
    sim.UpdateGimbalPertYPR(np.array((0, 0, ang4)))
    sim.UpdatePlots()
    [cang1, cang2, cang3, cang4] = sim.CalculateCriticalAngles(v_rg)  # right, left, top, bottom
    test = 1

    # # example of cycling through UAV gimbal perturbation orientations
    # for i in range(0, 40, 1):
    #     sim.UpdateGimbalPertYPR(np.array((np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))))
    #     sim.UpdatePlots()

    # # example of cycling through UAV positions
    # for i in range(-10, 20, 1):
    #     sim.UpdateX(np.array((i, 0, 40)))
    #     sim.UpdatePlots()
    # for j in range(0, 20, 1):
    #     sim.UpdateX(np.array((i, j, 40)))
    #     sim.UpdatePlots()
    # for k in range(40, 80, 1):
    #     sim.UpdateX(np.array((i, j, k)))
    #     sim.UpdatePlots()
    #
    # # example of cycling through target positions
    # for i in range(0, -10, -1):
    #     sim.UpdateTargetX(np.array((i, 0, 0)))
    #     sim.UpdatePlots()
    # for j in range(0, 10, 1):
    #     sim.UpdateTargetX(np.array((i, j, 0)))
    #     sim.UpdatePlots()
    # for k in range(0, 10, 1):
    #     sim.UpdateTargetX(np.array((i, j, k)))
    #     sim.UpdatePlots()
    #
    # # example of cycling through camera FOV angles
    # for i in range(45, 15, -1):
    #     sim.UpdateHFOV(np.deg2rad(i))
    #     sim.UpdatePlots()
    # for i in range(45, 20, -1):
    #     sim.UpdateVFOV(np.deg2rad(i))
    #     sim.UpdatePlots()

    plt.show()
