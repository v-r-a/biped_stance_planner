/* Copyright 2023 Vyankatesh Ashtekar

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstdio>
#include <cstring>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <GLFW/glfw3.h>
#include "../include/mujoco/mujoco.h"

// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// Frame selector
bool tf = true;  // IK torso frame
bool lf = false; // IK left foot frame
bool rf = false; // IK right foot frame

// Home position: standing on two feet upright
double txyz_rst[3] = {0.0000, -0.00, 0.21};
double tkphi_rst[4] = {1, 0, 0, 0};
double lfxyz_rst[3] = {0.033, -0.00, 0.00};
double lfkphi_rst[4] = {1, 0, 0, 0};
double rfxyz_rst[3] = {-0.033, -0.00, 0.00};
double rfkphi_rst[4] = {1, 0, 0, 0};

double txyz[3] = {0.0000, -0.00, 0.21};
double tkphi[4] = {1, 0, 0, 0};
double lfxyz[3] = {0.033, -0.00, 0.00};
double lfkphi[4] = {1, 0, 0, 0};
double rfxyz[3] = {-0.033, -0.00, 0.00};
double rfkphi[4] = {1, 0, 0, 0};

// Initial estimate for the joint angles of the robot in home position.
double qjans_rst[12] = {0.000, -0.000, -0.003, -0.012, -0.004, 0.000, 0.000, -0.000, -0.003, -0.012, -0.004, 0.000};
double qjans[12] = {-0.33051, -0.00203289, 0.86991, -1.34092, 0.429947, 0.0101976, 0.072782, -0.035326, 0.310979, -0.739273, 0.396155, 0.0522848};

double incr = 0.001;    // increment for site x y z movements in the IK utility
double incr_ang = 0.05; // increment for  rotations in the IK utility

// Drawing objects in scene
void drawWorldFrame(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom1, *mygeom2, *mygeom3;
    mjtNum scl = mm->stat.meansize;
    mjtNum arrowsize[3] = {0.005, 0.005, 1};
    mjtNum arrowpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    // Red X -- Green Y -- BLue Z
    float arrowrgbaX[4] = {0.9, 0., 0., 0.8};
    float arrowrgbaY[4] = {0, 0.9, 0., 0.8};
    float arrowrgbaZ[4] = {0., 0., 0.9, 0.8};
    mjtNum arrowscale = scl * 2;

    // Set Arrow length:
    arrowsize[2] = arrowscale;

    // Set Arrow direction:
    mjtNum diffX[3] = {1, 0, 0};
    mjtNum diffY[3] = {0, 1, 0};
    mjtNum diffZ[3] = {0, 0, 1};
    mjtNum myquatX[4] = {1, 0, 0, 0};
    mjtNum myquatY[4] = {1, 0, 0, 0};
    mjtNum myquatZ[4] = {1, 0, 0, 0};

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquatX, diffX);
    mju_quat2Mat(myrot3x3, myquatX);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom1 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom1->objtype = mjOBJ_UNKNOWN;
    mygeom1->objid = -1;
    mygeom1->category = mjCAT_DECOR;
    mygeom1->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom1, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3, arrowrgbaX);
    // mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    /*--------------------------------------------------------------------------------*/
    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquatY, diffY);
    mju_quat2Mat(myrot3x3, myquatY);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom2 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom2->objtype = mjOBJ_UNKNOWN;
    mygeom2->objid = -1;
    mygeom2->category = mjCAT_DECOR;
    mygeom2->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom2, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3, arrowrgbaY);
    // mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    /*--------------------------------------------------------------------------------*/
    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquatZ, diffZ);
    mju_quat2Mat(myrot3x3, myquatZ);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom3 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom3->objtype = mjOBJ_UNKNOWN;
    mygeom3->objid = -1;
    mygeom3->category = mjCAT_DECOR;
    mygeom3->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom3, mjGEOM_ARROW, arrowsize, arrowpos, myrot3x3, arrowrgbaZ);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
    /*--------------------------------------------------------------------------------*/
}

void drawCOM(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{
    // add a decorative geometry
    mjvGeom *mygeom;
    mjtNum sphsize[3] = {0.005, 0, 0};
    mjtNum sphpos[3] = {0., 0., 0.};
    mjtNum myrot3x3[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    float sphrgba[4] = {0.9, 0., 0., 0.9}; // Red

    mju_copy3(sphpos, dd->subtree_com);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom->objtype = mjOBJ_UNKNOWN;
    mygeom->objid = -1;
    mygeom->category = mjCAT_DECOR;
    mygeom->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom, mjGEOM_SPHERE, sphsize, sphpos, myrot3x3, sphrgba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

void drawCOMproj(mjModel *mm, mjData *dd, mjvScene *scene, const mjvOption *opt)
{

    //----------------------------------------------------------------------
    // add a line decorative geometry
    mjvGeom *mygeom2;
    mjtNum linesize[3] = {0., 0., -1};
    mjtNum linepos[3] = {0., 0., 0.};
    mjtNum com_proj[3];
    mjtNum diff[3];
    mjtNum myquat[4] = {1, 0, 0, 0};
    mjtNum myrot3x3_2[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    // Red
    float linergba[4] = {0.9, 0., 0., 0.9};

    // line starts at linepos
    mju_copy3(linepos, dd->subtree_com);
    com_proj[0] = dd->subtree_com[0];
    com_proj[1] = dd->subtree_com[1];
    // Line end: com_proj, line start: com
    mju_sub3(diff, com_proj, dd->subtree_com);
    // line length = height of COM from ground
    linesize[2] = mju_norm3(diff);

    // set mat to minimal rotation aligning b-a with z axis
    mju_quatZ2Vec(myquat, diff);
    mju_quat2Mat(myrot3x3_2, myquat);

    // one more geom to render
    scene->ngeom = scene->ngeom + 1;
    // mygeom now points to the last location in geoms buffer
    mygeom2 = scene->geoms + scene->ngeom - 1;
    // decor geom
    mygeom2->objtype = mjOBJ_UNKNOWN;
    mygeom2->objid = -1;
    mygeom2->category = mjCAT_DECOR;
    mygeom2->segid = scene->ngeom;

    // Add it to the scene
    mjv_initGeom(mygeom2, mjGEOM_LINE, linesize, linepos, myrot3x3_2, linergba);
    mjv_addGeoms(mm, dd, opt, NULL, mjCAT_DECOR, scene);
}

// Read-write functions (all files read from/ written to in ../bin directory)
void save_qpos(const mjModel *mm, mjData *dd)
{
    FILE *fid;
    fid = fopen("qpos_export.txt", "w");
    for (int i = 0; i < mm->nq; i++)
    {
        fprintf(fid, "%lg\n", dd->qpos[i]);
    }

    fclose(fid);
}

void load_qpos(const mjModel *mm, mjData *dd)
{
    FILE *fid;
    fid = fopen("qpos_export.txt", "r");
    int ferr = 0;
    if (fid != NULL)
    {
        for (int i = 0; i < mm->nq; i++)
        {
            ferr = fscanf(fid, "%lg\n", &(dd->qpos[i]));
            (!ferr) && (printf("error loading qpos from qpos_export.txt\n"));
        }

        fclose(fid);
    }
}

void save_sitepose(double ip_txyz[3], double ip_tkphi[4], double ip_lfxyz[3], double ip_lfkphi[4], double ip_rfxyz[3], double ip_rfkphi[4])
{
    FILE *fid;
    fid = fopen("site_export.txt", "w");

    // Torso site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_txyz[i]);
    // Torso k phi
    for (int i = 0; i < 4; i++)
        fprintf(fid, "%lg\n", ip_tkphi[i]);

    // Left foot site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_lfxyz[i]);
    // Left foot k phi
    for (int i = 0; i < 4; i++)
        fprintf(fid, "%lg\n", ip_lfkphi[i]);

    // Right foot site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_rfxyz[i]);
    // Right foot k phi
    for (int i = 0; i < 4; i++)
        fprintf(fid, "%lg\n", ip_rfkphi[i]);

    fclose(fid);
}

void save_com_pos(double ip_com_pos[3])
{
    FILE *fid;
    fid = fopen("com_export.txt", "w");

    // Torso site X Y Z
    for (int i = 0; i < 3; i++)
        fprintf(fid, "%lg\n", ip_com_pos[i]);
}

void load_sitepose(double op_txyz[3], double op_tkphi[4], double op_lfxyz[3], double op_lfkphi[4], double op_rfxyz[3], double op_rfkphi[4])
{
    extern double txyz[3], tkphi[4], lfxyz[3], lfkphi[4], rfxyz[3], rfkphi[4];
    FILE *fid;
    fid = fopen("site_export.txt", "r");
    int ferr = 0;

    if (fid != NULL)
    {
        // Torso site X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &txyz[i]);
        // Torso k phi
        for (int i = 0; i < 4; i++)
            ferr = fscanf(fid, "%lg\n", &tkphi[i]);

        // Left foot site X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &lfxyz[i]);
        // Left foot k phi
        for (int i = 0; i < 4; i++)
            ferr = fscanf(fid, "%lg\n", &lfkphi[i]);

        // Right foot site X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &rfxyz[i]);
        // Right foot k phi
        for (int i = 0; i < 4; i++)
            ferr = fscanf(fid, "%lg\n", &rfkphi[i]);

        fclose(fid);
    }
}

void load_com_pos(double op_com_pos[3])
{
    extern double com_pos[3];
    FILE *fid;
    fid = fopen("com_export.txt", "r");
    int ferr = 0;

    if (fid != NULL)
    {
        // Centre of mass X Y Z
        for (int i = 0; i < 3; i++)
            ferr = fscanf(fid, "%lg\n", &op_com_pos[i]);

        fclose(fid);
    }
}

// Supporting functions for the inverse kinematics function
void my_mju_quat2axisAngle(mjtNum *quat, mjtNum *axisa)
{
    // Create a copy of inputs for working
    double iquat[4];
    mju_copy4(iquat, quat);
    // mju_normalize4(iquat);
    double q0 = iquat[0], q1 = iquat[1], q2 = iquat[2], q3 = iquat[3];

    double temp, sinphiby2;

    // Find the angle
    sinphiby2 = mju_sqrt(q1 * q1 + q2 * q2 + q3 * q3);
    axisa[3] = atan2(sinphiby2, q0) * 2;

    // When q0 == +/-1, i.e., when phi == 0, there is no rotation. Axis could be anything. Setting it to {0,0,1}.
    if (abs(q0 * q0 - 1) < 1e-6)
    {
        axisa[0] = 0;
        axisa[1] = 0;
        axisa[2] = 1;
    }
    else
    {
        temp = mju_sqrt(1 - q0 * q0);
        axisa[0] = q1 / temp;
        axisa[1] = q2 / temp;
        axisa[2] = q3 / temp;
    }
}

void my_mju_kphi_equiv(double incr_k_phi[], double curr_k_phi[], double op_k_phi[])
{
    // through quaternions
    double q_curr[4], q_incr[4], q_op[4];
    mju_axisAngle2Quat(q_curr, curr_k_phi, curr_k_phi[3]);
    mju_axisAngle2Quat(q_incr, incr_k_phi, incr_k_phi[3]);
    mju_mulQuat(q_op, q_incr, q_curr);
    my_mju_quat2axisAngle(q_op, op_k_phi);

    // defining direction of new k
    // if (mju_sign(curr_k_phi[3]) != mju_sign(op_k_phi[3]))
    // {
    //     mju_scl(op_k_phi, op_k_phi, -1, 4); // Flip to -k -phi
    // }
}

double wrap_angles(double angle)
{
    double op;
    op = atan2(sin(angle), cos(angle));
    return op;
}

void my_mju_relQuat(mjtNum *qdif, const mjtNum qa[4], const mjtNum qb[4])
{
    // qdif = neg(qb)*qa
    mjtNum qneg[4];
    mju_negQuat(qneg, qb);
    mju_mulQuat(qdif, qneg, qa);
}

void bioloid_12dof_IK_position_v2(const mjModel *mm, int tsno, int lfsno, int rfsno,
                                  double global_torso[], double global_torso_kphi[],
                                  double global_leftfoot[], double global_leftfoot_kphi[],
                                  double global_rightfoot[], double global_rightfoot_kphi[],
                                  double qj_op[],
                                  int nitrmax, double ef, double ex)
{
    bool solconv = true;
    /*---------Processing Inputs---Convert axis angle to quat------------------------*/
    double global_torso_kaxis[3] = {global_torso_kphi[0], global_torso_kphi[1], global_torso_kphi[2]};
    double global_leftfoot_kaxis[3] = {global_leftfoot_kphi[0], global_leftfoot_kphi[1], global_leftfoot_kphi[2]};
    double global_rightfoot_kaxis[3] = {global_rightfoot_kphi[0], global_rightfoot_kphi[1], global_rightfoot_kphi[2]};
    mju_normalize3(global_torso_kaxis);
    mju_normalize3(global_leftfoot_kaxis);
    mju_normalize3(global_rightfoot_kaxis);
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_axisAngle2Quat(global_torso_quat, global_torso_kaxis, global_torso_kphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, global_leftfoot_kaxis, global_leftfoot_kphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, global_rightfoot_kaxis, global_rightfoot_kphi[3]);

    /*--make mjData instance for IK problem Jacobian computation----------------------*/
    mjData *ik_d = mj_makeData(mm);
    double z3[3] = {0., 0., 0.}, q_eye4[4] = {1, 0, 0, 0}, internal_qj_op[12];
    // Now the floating base is at origin with default orientation--->
    mju_copy3(ik_d->qpos, z3);
    mju_copy4(ik_d->qpos + 3, q_eye4);
    /*
    All calculations in Torso site frame. It has a constant offset w.r.t. the floating base
    left foot pos and orientation,
    right foot and orientation.
    */

    // Create copy of input estimate and update it. Copy back the answer at last if the iterations converge
    mju_copy(internal_qj_op, qj_op, 12);
    Eigen::Map<Eigen::VectorXd> eig_qj_op(internal_qj_op, 12);

    double target_lt_pos[3], target_rt_pos[3], T_target_lt_pos[3], T_target_rt_pos[3];
    double target_lt_quat[4], target_rt_quat[4];
    double R0T[9], RT0[9];
    mju_quat2Mat(R0T, global_torso_quat);
    mju_transpose(RT0, R0T, 3, 3);
    my_mju_relQuat(target_lt_quat, global_leftfoot_quat, global_torso_quat);  // Set desired orientation
    my_mju_relQuat(target_rt_quat, global_rightfoot_quat, global_torso_quat); // Set desied orientation
    mju_sub3(target_lt_pos, global_leftfoot, global_torso);
    mju_mulMatVec(T_target_lt_pos, RT0, target_lt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xl-Xt)
    mju_sub3(target_rt_pos, global_rightfoot, global_torso);
    mju_mulMatVec(T_target_rt_pos, RT0, target_rt_pos, 3, 3); // Set desired position in Torso frame: RT0(Xr-Xt)

    /*--Compute the linear velocity and angular velocity Jacobian matrices-------------*/
    double l_jacV[3 * 18], l_jacW[3 * 18], r_jacV[3 * 18], r_jacW[3 * 18];

    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacV(l_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> L_JacW(l_jacW, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacV(r_jacV, 3, 18);
    Eigen::Map<Eigen::Matrix<double, 3, 18, Eigen::RowMajor == 1>> R_JacW(r_jacW, 3, 18);
    Eigen::Matrix<double, 12, 12, Eigen::RowMajor == 1> J; // Matrix gradient of residual(qj)

    double residual[12]; // Equation: residual(qj) = 0 is being solved
    Eigen::Map<Eigen::VectorXd> eig_residual(residual, 12);

    double T_curr_lt_pos[3], T_curr_rt_pos[3], curr_lt_quat[4], curr_rt_quat[4];
    Eigen::VectorXd dqj_nm(12); // Output of Newton's method

    int itr = 0;
    double errorf = 1000, errorX = 1000;
    while (errorf > ef && errorX > ex && itr < nitrmax)
    {
        // Set the new qj for revised computations
        mju_copy(ik_d->qpos + 7, internal_qj_op, 12);
        // Compute mj_step1()
        mj_step1(mm, ik_d);
        // Compute Jacobian matrices in Torso frame
        mj_jacSite(mm, ik_d, l_jacV, l_jacW, lfsno);
        mj_jacSite(mm, ik_d, r_jacV, r_jacW, rfsno);

        // Read current position and orientation values (These values are in torso frame only becuase ik_d is set to torso frame)
        my_mju_relQuat(curr_lt_quat, ik_d->sensordata + 10, ik_d->sensordata + 3); // Orientation of left foot w.r.t. torso (Orientation of torso is not going to change!)
        my_mju_relQuat(curr_rt_quat, ik_d->sensordata + 17, ik_d->sensordata + 3); // Orientation of right foot w.r.t. torso (Orientation of torso is not going to change!)
        mju_sub3(T_curr_lt_pos, ik_d->sensordata + 7, ik_d->sensordata);           // T(Xl - Xt) (Xt is not going to change!)
        mju_sub3(T_curr_rt_pos, ik_d->sensordata + 14, ik_d->sensordata);          // T(Xr - Xt) (Xt is not going to change!)

        // Update residual
        mju_sub3(residual, T_target_lt_pos, T_curr_lt_pos);      // compute residual
        mju_subQuat(residual + 3, target_lt_quat, curr_lt_quat); // in tgt space
        mju_sub3(residual + 6, T_target_rt_pos, T_curr_rt_pos);  // compute residual
        mju_subQuat(residual + 9, target_rt_quat, curr_rt_quat); // in tgt space

        // Update big J
        J.block<3, 12>(0, 0) = L_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(3, 0) = L_JacW.block<3, 12>(0, 6);
        J.block<3, 12>(6, 0) = R_JacV.block<3, 12>(0, 6);
        J.block<3, 12>(9, 0) = R_JacW.block<3, 12>(0, 6);

        // Solve
        dqj_nm = J.fullPivHouseholderQr().solve(eig_residual);
        // dqj_nm.block<6,1>(0, 0) = J.block<6, 6>(0, 0).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(0, 0));
        // dqj_nm.block<6,1>(6, 0) = J.block<6, 6>(6, 6).fullPivHouseholderQr().solve(eig_residual.block<6, 1>(6, 0));

        // Compute error norms for early termination
        errorf = eig_residual.norm();
        errorX = dqj_nm.norm();

        // Update qj
        eig_qj_op += dqj_nm;

        itr++;
    }

    // Convergence notification
    if (errorf > ef && errorX > ex)
    {
        solconv = false;
        printf("Sol not converged. Errorf: %lg \t ErrorQ: %lg \t Nitr: %d\n", errorf, errorX, itr);
    }
    else
    {
        mju_copy(qj_op, internal_qj_op, 12);
    }

    // Get the angles back in domain -pi to pi
    // May be problematic?
    for (int i = 0; i < 12; i++)
        qj_op[i] = wrap_angles(qj_op[i]);

    mj_deleteData(ik_d);
}

void pose_updater(const mjModel *mm, mjData *dd)
{
    // qjans is updated only if solution converges
    bioloid_12dof_IK_position_v2(mm, 0, 1, 2, txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi, qjans, 20, 1e-6, 1e-5);

    // update the pose--------------------------------------------------
    double global_torso_quat[4], global_leftfoot_quat[4], global_rightfoot_quat[4];
    mju_normalize3(tkphi);
    mju_normalize3(lfkphi);
    mju_normalize3(rfkphi);
    mju_axisAngle2Quat(global_torso_quat, tkphi, tkphi[3]);
    mju_axisAngle2Quat(global_leftfoot_quat, lfkphi, lfkphi[3]);
    mju_axisAngle2Quat(global_rightfoot_quat, rfkphi, rfkphi[3]);

    // Position of the site w.r.t the floating base
    int tsno = 0; // torso site id
    double torso_site_offsetpos[3], torso_site_offsetquat[4];
    mju_copy3(torso_site_offsetpos, mm->site_pos + tsno * 3);   // Correct
    mju_copy4(torso_site_offsetquat, mm->site_quat + tsno * 4); // Correct

    // Floating base orientation
    my_mju_relQuat(dd->qpos + 3, global_torso_quat, torso_site_offsetquat); // Correct

    // Floating base position
    double rot_fb[9];
    mju_quat2Mat(rot_fb, dd->qpos + 3);
    mju_mulMatVec(dd->qpos, rot_fb, torso_site_offsetpos, 3, 3);
    mju_sub3(dd->qpos, txyz, dd->qpos);

    //  Rest of the joints qj = [qjL|qjR]
    mju_copy(dd->qpos + 7, qjans, 12);
}

// keyboard callback
void bioloid_keyboard_IK_util(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    double quat_temp[4], quat_current[4], xaxis[3] = {1, 0, 0}, yaxis[3] = {0, 1, 0}, zaxis[3] = {0, 0, 1}; // For the rotations

    if (act = GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_F:
            // Contact force
            opt.flags[mjVIS_CONTACTFORCE] = true;
            break;

        case GLFW_KEY_T:
            // Transparency
            opt.flags[mjVIS_TRANSPARENT] = true;
            break;

        case GLFW_KEY_S:
            // Site frame visualization
            opt.frame = mjFRAME_SITE;
            break;

        case GLFW_KEY_C:
            // Contact frame visualization
            opt.frame = mjFRAME_CONTACT;
            break;

        case GLFW_KEY_J:
            opt.flags[mjVIS_JOINT] = true;
            break;

        case GLFW_KEY_L:
            opt.label = mjLABEL_JOINT;
            break;

        case GLFW_KEY_1:
            // Choose torso site
            tf = true;
            lf = false;
            rf = false;
            printf("Torso site selected\n");
            break;

        case GLFW_KEY_2:
            // Choose left foot site
            tf = false;
            lf = true;
            rf = false;
            printf("Left foot site selected\n");
            break;

        case GLFW_KEY_3:
            // Choose right foot site
            tf = false;
            lf = false;
            rf = true;
            printf("Right foot sire selected\n");
            break;

        case GLFW_KEY_UP:
            // Y coord ++
            (tf && (txyz[1] += incr));
            (lf && (lfxyz[1] += incr));
            (rf && (rfxyz[1] += incr));
            break;

        case GLFW_KEY_DOWN:
            // Y coord --
            (tf && (txyz[1] -= incr));
            (lf && (lfxyz[1] -= incr));
            (rf && (rfxyz[1] -= incr));
            break;

        case GLFW_KEY_RIGHT:
            // X coord ++
            (tf && (txyz[0] += incr));
            (lf && (lfxyz[0] += incr));
            (rf && (rfxyz[0] += incr));
            break;

        case GLFW_KEY_LEFT:
            // X coord --
            (tf && (txyz[0] -= incr));
            (lf && (lfxyz[0] -= incr));
            (rf && (rfxyz[0] -= incr));
            break;

        case GLFW_KEY_PAGE_UP:
            // Z coord ++
            (tf && (txyz[2] += incr));
            (lf && (lfxyz[2] += incr));
            (rf && (rfxyz[2] += incr));
            break;

        case GLFW_KEY_PAGE_DOWN:
            // Z coord --
            (tf && (txyz[2] -= incr));
            (lf && (lfxyz[2] -= incr));
            (rf && (rfxyz[2] -= incr));
            break;

        case GLFW_KEY_4:
            // Pitch ccw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, xaxis, incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);

                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_5:
            // Pitch cw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, xaxis, -incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }

            break;
        case GLFW_KEY_6:
            // Roll ccw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, yaxis, incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_7:
            // Roll cw rotation

            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, yaxis, -incr_ang);

            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;
        case GLFW_KEY_8:
            // Yaw ccw rotation
            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, zaxis, incr_ang);
            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_9:
            // Yaw cw rotation
            // Incremental rotation
            mju_axisAngle2Quat(quat_temp, zaxis, -incr_ang);
            if (tf)
            {
                // get the quaternion corresponding to the current kphi
                mju_axisAngle2Quat(quat_current, tkphi, tkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                // set the total new kphi
                my_mju_quat2axisAngle(quat_temp, tkphi);
            }
            if (lf)
            {
                mju_axisAngle2Quat(quat_current, lfkphi, lfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, lfkphi);
            }
            if (rf)
            {
                mju_axisAngle2Quat(quat_current, rfkphi, rfkphi[3]);
                // mju_mulQuat(quat_temp, quat_temp, quat_current);
                // post multiply for body fixed rotations
                mju_mulQuat(quat_temp, quat_current, quat_temp);
                my_mju_quat2axisAngle(quat_temp, rfkphi);
            }
            break;

        case GLFW_KEY_W:
            // Write qpos to file
            printf("Saving qpos to file: qpos_export.txt\n");
            save_qpos(m, d);
            printf("Saving site frame data to site_export.txt\n");
            save_sitepose(txyz, tkphi, lfxyz, lfkphi, rfxyz, rfkphi);
            printf("Saving COM position to com_export.txt\n");
            save_com_pos(d->subtree_com);
            break;

        case GLFW_KEY_BACKSPACE:
            // backspace: reset simulation
            mj_resetData(m, d);
            // load qpos0 given in the bioloid_IK_util.cc code and not the file.
            mju_copy3(txyz, txyz_rst);
            mju_copy3(lfxyz, lfxyz_rst);
            mju_copy3(rfxyz, rfxyz_rst);
            mju_copy4(tkphi, tkphi_rst);
            mju_copy4(lfkphi, lfkphi_rst);
            mju_copy4(rfkphi, rfkphi_rst);
            mju_copy(qjans, qjans_rst, 12);

            // Intentional fall through this case to the following case:

        case GLFW_KEY_ESCAPE:
            // Reset stuff
            opt.flags[mjVIS_CONTACTFORCE] = false;
            opt.flags[mjVIS_TRANSPARENT] = false;
            opt.frame = mjFRAME_NONE;
            opt.flags[mjVIS_JOINT] = false;
            opt.label = mjLABEL_NONE;
            break;
        }
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
    {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
    {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// main function
int main(int argc, const char **argv)
{
    // check command-line arguments
    if (argc != 2)
    {
        std::printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    // load and compile model
    char error[1000] = "Could not load binary model";
    if (std::strlen(argv[1]) > 4 && !std::strcmp(argv[1] + std::strlen(argv[1]) - 4, ".mjb"))
    {
        m = mj_loadModel(argv[1], 0);
    }
    else
    {
        m = mj_loadXML(argv[1], 0, error, 1000);
    }
    if (!m)
    {
        mju_error_s("Load model error: %s", error);
    }

    // make data
    d = mj_makeData(m);

    // init GLFW
    if (!glfwInit())
    {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, bioloid_keyboard_IK_util);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // Venky edits
    scn.flags[mjRND_SHADOW] = false;

    //  run main loop, keyboard interrupt based robot position update
    while (!glfwWindowShouldClose(window))
    {
        pose_updater(m, d);
        mj_forward(m, d);

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // Decorative geometries
        drawWorldFrame(m, d, &scn, &opt);
        drawCOM(m, d, &scn, &opt);
        drawCOMproj(m, d, &scn, &opt);

        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

// terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
}
