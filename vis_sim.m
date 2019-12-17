% UAV Visual Constraints Animation
% Hayden Morgan
% 11/22/19

format short
clear all
close all

% UAV and target conditions
r = 10;
h = 40;
x = [-r; 0; h];
rpy = [20,10,10];
rpyg = [-20,-10,-10];
xt = [0; 0; 0];

phi_fov = deg2rad(45);
phi_g = deg2rad(rpyg(1));
phir = phi_fov/2-phi_g+atan2(sign(x(1))*r,h); % right critical constraint
phil = -phi_fov/2-phi_g+atan2(sign(x(1))*r,h); % left critical constraint

phi = deg2rad(rpy(1));

theta_fov = deg2rad(45);
theta_g = deg2rad(rpyg(2));
thetab = theta_fov/2-theta_g; % back critical constraint
thetaf = -theta_fov/2-theta_g; % front critical constraint

theta = deg2rad(rpy(2));

psi = deg2rad(rpy(3));
psi_g = deg2rad(rpyg(3));

xlos = h*tan(-phi-phi_g)
rx = (xlos + h*tan(phi_fov/2))/(1-xlos*tan(phi_fov/2)/h)
lx = (xlos - h*tan(phi_fov/2))/(1+xlos*tan(phi_fov/2)/h)

ylos = h*tan(theta+theta_g)
by = (ylos + h*tan(theta_fov/2))/(1-ylos*tan(theta_fov/2)/h)
fy = (ylos - h*tan(theta_fov/2))/(1+ylos*tan(theta_fov/2)/h)

% general plotting
linedist = 100;
endpoint = [0;
            -linedist];
plot2ddata = false;
plot3ddata = true;

% ######################################################################
% #############################3D PLOT##################################
% ######################################################################

if plot3ddata
    
    % setup background base
    figure(2)
    los = [x, xt];
    plot3(los(1,:),los(2,:),los(3,:));
    hold on
    pose_point = plot3(x(1),x(2),x(3), 'ko', 'MarkerFaceColor', 'k');
    pose_line = plot3([x(1),x(1)], [x(2),x(2)], [x(3),0], 'k--');
    pose_target = plot3(xt(1),xt(2),xt(3), 'ro', 'MarkerFaceColor', 'r');
    lfov = [0, 0;
            0, 0;
            0, 200;];
    lquiv = [0, 10;
             0, 0;
             0, 0;];
    lcamx = [0, 10;
             0, 0;
             0, 0;];
    lbNorth = [0, 10;
             0, 0;
             0, 0;];
    rng = 100;
    xlim([-rng,rng])
    ylim([-rng,rng])
    zlim([-2,rng])
    title('3D Sim')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    grid on
    pbaspect([1 1 1])
    hold on
    
    Riv = [0, 1, 0;
           1, 0, 0;
           0, 0, -1];
    R_ib = rot3dzp(psi)*rot3dyp(theta)*rot3dxp(phi)*Riv;
    R_vb = rot3dzp(psi)*rot3dyp(theta)*rot3dxp(phi);
    R_bg = rot3dxp(phi_g)*rot3dyp(theta_g)*rot3dzp(psi_g);
    R_gc = eye(3);
    R_cfov1_x = rot3dyp(phi_fov/2);
    R_cfov2_x = rot3dyp(-phi_fov/2);
    R_cfov1_y = rot3dxp(theta_fov/2);
    R_cfov2_y = rot3dxp(-theta_fov/2);
    
    % Rotate animation lines back out to common inertial frame for plotting

    loptax = R_ib'*R_bg'*lfov + x;
    lfov1 = R_ib'*R_bg'*R_gc'*R_cfov1_y'*R_cfov1_x'*lfov + x;
    lfov2 = R_ib'*R_bg'*R_gc'*R_cfov2_y'*R_cfov1_x'*lfov + x;
    lfov3 = R_ib'*R_bg'*R_gc'*R_cfov1_y'*R_cfov2_x'*lfov + x;
    lfov4 = R_ib'*R_bg'*R_gc'*R_cfov2_y'*R_cfov2_x'*lfov + x;
    lquiv = R_ib'*lquiv + x;
    lcamx = R_ib'*R_bg'*R_gc'*lcamx + x;
    lbNorth = Riv'*lbNorth + x;
    lbz = R_ib'*lfov + x;
    
    ploptax = plot3(loptax(1,:), loptax(2,:), loptax(3,:), 'r--');
    plfov1 = plot3(lfov1(1,:), lfov1(2,:), lfov1(3,:), 'r-');
    plfov2 = plot3(lfov2(1,:), lfov2(2,:), lfov2(3,:), 'r-');
    plfov3 = plot3(lfov3(1,:), lfov3(2,:), lfov3(3,:), 'r-');
    plfov4 = plot3(lfov4(1,:), lfov4(2,:), lfov4(3,:), 'r-');
    plcamx = plot3(lcamx(1,:), lcamx(2,:), lcamx(3,:), 'r-');
    plbNorth = plot3(lbNorth(1,:), lbNorth(2,:), lbNorth(3,:), 'g-');
    plbz = plot3(lbz(1,:), lbz(2,:), lbz(3,:), 'k-');
    quiver3(lquiv(1,1),lquiv(2,1),lquiv(3,1),lquiv(1,2)-lquiv(1,1),lquiv(2,2)-lquiv(2,1),lquiv(3,2)-lquiv(3,1),'b','LineWidth',5)

    a=1;

end

% ######################################################################
% #############################2D PLOT##################################
% ######################################################################

% -------------------------ROLL-------------------------
if plot2ddata
    % setup background base
    figure(1)
    subplot(121)
    xlim([-20,20])
    ylim([-2, 50])
    line([-20,20], [0,0])
    title('Roll Visual Field')
    ylabel('z')
    xlabel('x')
    pbaspect([1 1 1])
    hold on
    pose_point = plot(x(1),x(3), 'ko', 'MarkerFaceColor', 'k');
    pose_line = plot([x(1),x(1)], [x(3),0], 'k--');
    
    R_ib_phi = rot2dp(phi); % roll inertial-body
    R_bc_phi = rot2dp(phi_g); % roll body-camera
    R_cfov1_phi = rot2dp(phi_fov/2);
    R_cfov2_phi = rot2dp(-phi_fov/2);

    % target plot
    plot(xt(1),0, 'ro', 'MarkerFaceColor', 'r')

    % roll animation
    br1 = R_cfov1_phi*R_bc_phi*R_ib_phi*endpoint + [x(1);x(3)];
    lr1 = plot([x(1),br1(1)], [x(3),br1(2)], 'r-');     % left fov
    br2 = R_cfov2_phi*R_bc_phi*R_ib_phi*endpoint + [x(1);x(3)];
    lr2 = plot([x(1),br2(1)], [x(3),br2(2)], 'r-');     % right fov
    br3 = R_ib_phi*endpoint + [x(1);x(3)];
    lr3 = plot([x(1),br3(1)], [x(3),br3(2)], 'k-');    % pose line
    br4 = R_bc_phi*R_ib_phi*endpoint + [x(1);x(3)];
    lr4 = plot([x(1),br4(1)], [x(3),br4(2)], 'r--');    % optical axis line

    % % Roll animation
    % for i = -20:1:20
    %     phi = deg2rad(i);
    % 
    %     R_ib_phi = rot2dp(phi);
    %     R_bc_phi = rot2dp(phi_g);
    %     R_cfov1_phi = rot2dp(phi_fov/2);
    %     R_cfov2_phi = rot2dp(-phi_fov/2);
    %     
    %     br1 = R_cfov1_phi*R_bc_phi*R_ib_phi*endpoint;
    %     lr1.XData = [x(1),br1(1)];
    %     lr1.YData = [x(3),br1(2)];
    %     
    %     br2 = R_cfov2_phi*R_bc_phi*R_ib_phi*endpoint;
    %     lr2.XData = [x(1),br2(1)];
    %     lr2.YData = [x(3),br2(2)];
    %     
    %     br3 = R_ib_phi*endpoint;
    %     lr3.XData = [x(1),br3(1)];
    %     lr3.YData = [x(3),br3(2)];
    %     
    %     br4 = R_bc_phi*R_ib_phi*endpoint;
    %     lr4.XData = [x(1),br4(1)];
    %     lr4.YData = [x(3),br4(2)];
    %     
    %     refreshdata
    %     pause(0.02)
    % end
    % 
    % % Roll gimball animation
    % for i = -20:1:20
    %     phi_g = deg2rad(i);
    % 
    %     R_ib_phi = rot2dp(phi);
    %     R_bc_phi = rot2dp(phi_g);
    %     R_cfov1_phi = rot2dp(phi_fov/2);
    %     R_cfov2_phi = rot2dp(-phi_fov/2);
    %     
    %     br1 = R_cfov1_phi*R_bc_phi*R_ib_phi*endpoint;
    %     lr1.XData = [x(1),br1(1)];
    %     lr1.YData = [x(3),br1(2)];
    %     
    %     br2 = R_cfov2_phi*R_bc_phi*R_ib_phi*endpoint;
    %     lr2.XData = [x(1),br2(1)];
    %     lr2.YData = [x(3),br2(2)];
    %     
    %     br3 = R_ib_phi*endpoint;
    %     lr3.XData = [x(1),br3(1)];
    %     lr3.YData = [x(3),br3(2)];
    %     
    %     br4 = R_bc_phi*R_ib_phi*endpoint;
    %     lr4.XData = [x(1),br4(1)];
    %     lr4.YData = [x(3),br4(2)];
    %     
    %     refreshdata
    %     pause(0.02)
    % end

    hold off

    % -------------------------PITCH-------------------------

    % setup background base
    subplot(122)
    xlim([-20,20])
    ylim([-2, 50])
    line([-20,20], [0,0])
    title('Pitch Visual Field')
    ylabel('z')
    xlabel('y')
    pbaspect([1 1 1])
    hold on
    quiver(x(2)+2,x(3),-3,0,0,'LineWidth',5)
    pose_point = plot(x(2),x(3), 'ko', 'MarkerFaceColor', 'k');
    pose_line = plot([x(2),x(2)], [x(3),0], 'k--');
    
    R_ib_theta = rot2dp(theta);
    R_bc_theta = rot2dp(theta_g);
    R_cfov1_theta = rot2dp(theta_fov/2);
    R_cfov2_theta = rot2dp(-theta_fov/2);

    % target plot
    plot(xt(2),0, 'ro', 'MarkerFaceColor', 'r')

    % pitch animation
    bp1 = R_cfov1_theta*R_bc_theta*R_ib_theta*endpoint + [x(2);x(3)];
    lp1 = plot([x(2),bp1(1)], [x(3),bp1(2)], 'b-');     % left fov
    bp2 = R_cfov2_theta*R_bc_theta*R_ib_theta*endpoint + [x(2);x(3)];
    lp2 = plot([x(2),bp2(1)], [x(3),bp2(2)], 'b-');     % right fov
    bp3 = R_ib_theta*endpoint + [x(2);x(3)];
    lp3 = plot([x(2),bp3(1)], [x(3),bp3(2)], 'k-');    % pose line      
    bp4 = R_bc_theta*R_ib_theta*endpoint + [x(2);x(3)];
    lp4 = plot([x(2),bp4(1)], [x(3),bp4(2)], 'b--');        % optical axis line

    % % pitch animation
    % for i = -20:1:20
    %     theta = deg2rad(i);
    % 
    %     R_ib_theta = rot2dp(theta);
    %     R_bc_theta = rot2dp(theta_g);
    %     R_cfov1_theta = rot2dp(theta_fov/2);
    %     R_cfov2_theta = rot2dp(-theta_fov/2);
    %     
    %     bp1 = R_cfov1_theta*R_bc_theta*R_ib_theta*endpoint;
    %     lp1.XData = [x(1),bp1(1)];
    %     lp1.YData = [x(3),bp1(2)];
    %     
    %     bp2 = R_cfov2_theta*R_bc_theta*R_ib_theta*endpoint;
    %     lp2.XData = [x(1),bp2(1)];
    %     lp2.YData = [x(3),bp2(2)];
    %     
    %     bp3 = R_ib_theta*endpoint;
    %     lp3.XData = [x(1),bp3(1)];
    %     lp3.YData = [x(3),bp3(2)];
    %     
    %     bp4 = R_bc_theta*R_ib_theta*endpoint;
    %     lp4.XData = [x(1),bp4(1)];
    %     lp4.YData = [x(3),bp4(2)];
    %     
    %     refreshdata
    %     pause(0.02)
    % end
    % 
    % % pitch gimbal animation
    % for i = -20:1:20
    %     theta_g = deg2rad(i);
    % 
    %     R_ib_theta = rot2dp(theta);
    %     R_bc_theta = rot2dp(theta_g);
    %     R_cfov1_theta = rot2dp(theta_fov/2);
    %     R_cfov2_theta = rot2dp(-theta_fov/2);
    %     
    %     bp1 = R_cfov1_theta*R_bc_theta*R_ib_theta*endpoint;
    %     lp1.XData = [x(1),bp1(1)];
    %     lp1.YData = [x(3),bp1(2)];
    %     
    %     bp2 = R_cfov2_theta*R_bc_theta*R_ib_theta*endpoint;
    %     lp2.XData = [x(1),bp2(1)];
    %     lp2.YData = [x(3),bp2(2)];
    %     
    %     bp3 = R_ib_theta*endpoint;
    %     lp3.XData = [x(1),bp3(1)];
    %     lp3.YData = [x(3),bp3(2)];
    %     
    %     bp4 = R_bc_theta*R_ib_theta*endpoint;
    %     lp4.XData = [x(1),bp4(1)];
    %     lp4.YData = [x(3),bp4(2)];
    %     
    %     refreshdata
    %     pause(0.02)
    % end

    hold off
end

a = rot3dxp(1)
b = rot3dyp(1)
c = rot3dzp(1)

A = rot3dxp(-1)
B = rot3dyp(-1)
C = rot3dzp(-1)
tes = 1

function R = rot2da(ang)
    R = [cos(ang), -sin(ang);
         sin(ang), cos(ang)];
end

function R = rot2dp(ang)
    R = rot2da(ang)';
end

function R = rot3dxp(ang)
    R = [1, 0, 0;
         0, cos(ang), -sin(ang);
         0, sin(ang), cos(ang)]';
end

function R = rot3dyp(ang)
    R = [cos(ang),  0,   sin(ang);
         0,         1,   0;
         -sin(ang), 0,   cos(ang)]';
end

function R = rot3dzp(ang)
    R = [cos(ang), -sin(ang), 0;
         sin(ang), cos(ang),  0;
         0,        0,         1]';
end


