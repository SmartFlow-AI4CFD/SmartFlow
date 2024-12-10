//=========================================
//PARAMETERS
triDmesh = 1; //Extrude?
D = 1; //Cylinder Diameter 
p = 4; //Mesh order

//Cylinder center 
x_center = 0;
y_center = 0;

//Domain sizes
x_min = -7.5*D;
x_max = 22.5*D;
y_min = -7.5*D;
y_max = 7.5*D;

Lz = 4*D;   //Span width

nz_env = 2; //Number of spanwise elements (first order) per pseudo-environment
n_psenv = 10; //Number of pseudo-environments in the domain (so that nz = nz_env * n_psenv)

//Consider buffer sizes
buffer_e_size = 5.0;
buffer_w_size = 2.5;
buffer_n_size = 2.5;
buffer_s_size = 2.5;

//Aprox. dist from walls to infl. layer limits
d = 1; 
  
//Random size to initialise (not important)
Size = 0.1; 

//Set mesh order
Mesh.ElementOrder = p;


//=========================================
//DOMAIN POINTS (considering the extra domain of the buffer)
Point(1) = {x_min - buffer_w_size, y_min - buffer_s_size, 0, Size};
Point(2) = {x_max + buffer_e_size, y_min - buffer_s_size, 0, Size};
Point(3) = {x_max + buffer_e_size, y_max + buffer_n_size, 0, Size};
Point(4) = {x_min - buffer_w_size, y_max + buffer_n_size, 0, Size};

//CYLINDER CENTER 
Point(5) = {x_center,y_center,0,Size};

//CYLINDER 45º POINTS
Point(6) = {x_center + (D/2)*Cos(45*Pi/180), y_center + (D/2)*Sin(45*Pi/180), 0,Size};
Point(7) = {x_center + (D/2)*Cos(135*Pi/180), y_center + (D/2)*Sin(135*Pi/180), 0,Size};
Point(8) = {x_center + (D/2)*Cos(225*Pi/180), y_center + (D/2)*Sin(225*Pi/180), 0,Size};
Point(9) = {x_center + (D/2)*Cos(315*Pi/180), y_center + (D/2)*Sin(315*Pi/180), 0,Size};

//INFLATION LAYER DIAGONAL POINTS
Point(10) = {x_center-d, y_center-d, 0, Size};
Point(11) = {x_center+d, y_center-d, 0, Size};
Point(12) = {x_center+d, y_center+d, 0, Size};
Point(13) = {x_center-d, y_center+d, 0, Size};

//INFLATION LAYER CIRCLES CENTER
Point(14) = {x_center, y_center+d, 0, Size};
Point(15) = {x_center-d, y_center, 0, Size};
Point(16) = {x_center, y_center-d, 0, Size};
Point(17) = {x_center+d, y_center, 0, Size};

//INFLATION LAYER DOMAIN POINTS 
Point(18) = {x_center-d, y_min - buffer_s_size, 0, Size};
Point(19) = {x_center+d, y_min - buffer_s_size, 0, Size};

Point(20) = {x_max + buffer_e_size, y_center-d, 0, Size};
Point(21) = {x_max + buffer_e_size, y_center+d, 0, Size};

Point(22) = {x_center-d, y_max + buffer_n_size, 0, Size};
Point(23) = {x_center+d, y_max + buffer_n_size, 0, Size};

Point(24) = {x_min - buffer_w_size, y_center-d, 0, Size};
Point(25) = {x_min - buffer_w_size, y_center+d, 0, Size};

Point(26) = {x_center,y_center+0.5*D+0.25*D,0,Size}; //BL Height aprox.
Point(27) = {x_max, y_max,0,Size}; //Domain without Buffers
Point(28) = {x_min, y_max,0,Size};
Point(29) = {x_min, y_min,0,Size};
Point(30) = {x_max, y_min,0,Size};

//=======================================
//DOMAIN CURVES 
Line(1) = {1,18};
Line(2) = {18,19};
Line(3) = {19,2};
Line(4) = {2,20};
Line(5) = {20,21};
Line(6) = {21,3};
Line(7) = {3,23};
Line(8) = {23,22};
Line(9) = {22,4};
Line(10) = {4,25};
Line(11) = {25,24};
Line(12) = {24,1};

//CYLINDER CURVES
Circle(13) = {6,5,7};
Circle(14) = {7,5,8};
Circle(15) = {8,5,9};
Circle(16) = {9,5,6};

//INFL. LAYER x+ CURVES
Circle(17) = {12,16,13};
Circle(18) = {13,17,10};
Circle(19) = {10,14,11};
Circle(20) = {11,15,12};

//INFL. LAYER y+ LINES 
Line(21) = {6,12};
Line(22) = {7,13};
Line(23) = {8,10};
Line(24) = {9,11};

//INFL. LAYER y+ LINES DOMAIN 
Line(25) = {12,23};
Line(26) = {13,22};
Line(27) = {13,25};
Line(28) = {10,24};
Line(29) = {10,18};
Line(30) = {11,19};
Line(31) = {11,20};
Line(32) = {12,21};

//Circle(33) = {26,5,26};

//=========================================
Line Loop(1) = {13,22,-17,-21};
Plane Surface (1) = {1};

Line Loop(2) = {14,23,-18,-22};
Plane Surface (2) = {2};

Line Loop(3) = {15,24,-19,-23};
Plane Surface (3) = {3};

Line Loop(4) = {16,21,-20,-24};
Plane Surface (4) = {4};

Line Loop(5) = {17,26,-8,-25};
Plane Surface(5) = {5};

Line Loop(6) = {18,28,-11,-27};
Plane Surface(6) = {6};

Line Loop(7) = {19,30,-2,-29};
Plane Surface(7) = {7};

Line Loop(8) = {20,32,-5,-31};
Plane Surface(8) = {8};

Line Loop(9) = {9,10,-27,26};
Plane Surface(9) = {9};

Line Loop(10) = {12,1,-29,28};
Plane Surface(10) = {10};

Line Loop(11) = {3,4,-31,30};
Plane Surface(11) = {11};

Line Loop(12) = {6,7,-25,32};
Plane Surface(12) = {12};


//========================================
Transfinite Line{21,22,23,24} = 8 Using Progression 1.1;  //y+ cylinder (Ay=0.02)
Transfinite Line{13,17,14,18,15,19,16,20,8,11,2,5} = 10;    //x+ cylinder (Ax=0.025)

Transfinite Line{25,26,27,28,29,30,9,-10,12,-1,-4,6} = 21 Using Progression 1.1;   //y+ domain
Transfinite Line{3,31,32,-7} = 32 Using Progression 1.1;   //y+ domain downstream


//===========================================
Transfinite Surface{1,2,3,4,5,6,7,8,9,10,11,12};
Recombine Surface{1,2,3,4,5,6,7,8,9,10,11,12};


//===========================================
If (triDmesh)

eps = 1e-3; //Small values required to search surfaces

// EXTRUSION
//Lenght of each pseudo-environment
L_psenv = Lz / n_psenv;

//Number of surfaces (Number of surfaces in the x-y plane defined above)
n_surf = 12;

//Loop every pseudo-env. and then every surface on it to make extrusions one by one
For i In {0:n_psenv-1} //Number of pseudo-environments
    For j In {0:n_surf-1} //Number of surfaces to extrude in every pseudo-environment

    //Extrusion of the first pseudo-environment Surface{1,2,3,4,5,...,n_surf}
    If (i==0)
        vol[] = Extrude {0, 0, L_psenv}{
            Surface{j+1};
            Layers{nz_env};
            Recombine;
        };

    //Extrusion of the remaining pseudo-environments
    Else
        vol[] = Extrude {0, 0, L_psenv}{
            Surface{surf_TOP_list[(i-1)*n_surf+j]};
            Layers{nz_env};
            Recombine;
        };
    EndIf

    //Retireve the ID number of the "top" surface of the extrusion --> Required in follow extrusion
    surf_TOP_list[i*n_surf+j] = vol[0];
    
    EndFor
EndFor


//SET PHYSICAL SURFACES/VOLUMES NAMES
//Assign WALL surfaces (One per pseudo-environment)
For i In {0:n_psenv-1}
    xmin_wall = x_center-D/2; xmax_wall = x_center+D/2; 
    ymin_wall = y_center-D/2; ymax_wall = y_center+D/2; 
    zmin_wall = L_psenv*i; zmax_wall = L_psenv*(i+1);
    wall_list() = Surface In BoundingBox {xmin_wall-eps, ymin_wall-eps, zmin_wall-eps,
                                            xmax_wall+eps, ymax_wall+eps, zmax_wall+eps};

    Physical Surface( Sprintf("wall-%g", i) ) = {wall_list()};
EndFor

//Assign INLET surfaces
xmin_in = x_min-buffer_w_size; xmax_in = x_min-buffer_w_size; 
ymin_in = y_min-buffer_s_size; ymax_in = y_max+buffer_n_size; 
zmin_in = 0; zmax_in = Lz;
inlet_list() = Surface In BoundingBox {xmin_in-eps, ymin_in-eps, zmin_in-eps,
                                       xmax_in+eps, ymax_in+eps, zmax_in+eps};
Physical Surface("inlet") = {inlet_list()};

//Assign OUTLET surfaces
xmin_out = x_max+buffer_e_size; xmax_out = x_max+buffer_e_size; 
ymin_out = y_min-buffer_s_size; ymax_out = y_max+buffer_n_size; 
zmin_out = 0; zmax_out = Lz;
outlet_list() = Surface In BoundingBox {xmin_out-eps, ymin_out-eps, zmin_out-eps,
                                        xmax_out+eps, ymax_out+eps, zmax_out+eps};
Physical Surface("outlet") = {outlet_list()};

//Assign TOP surfaces
xmin_top = x_min-buffer_w_size; xmax_top = x_max+buffer_e_size; 
ymin_top = y_max+buffer_n_size; ymax_top = y_max+buffer_n_size; 
zmin_top = 0; zmax_top = Lz;
top_list() = Surface In BoundingBox {xmin_top-eps, ymin_top-eps, zmin_top-eps,
                                     xmax_top+eps, ymax_top+eps, zmax_top+eps};
Physical Surface("top") = {top_list()};

//Assign BOT surfaces
xmin_bot = x_min-buffer_w_size; xmax_bot = x_max+buffer_e_size; 
ymin_bot = y_min-buffer_s_size; ymax_bot = y_min-buffer_s_size; 
zmin_bot = 0; zmax_bot = Lz;
bot_list() = Surface In BoundingBox {xmin_bot-eps, ymin_bot-eps, zmin_bot-eps,
                                     xmax_bot+eps, ymax_bot+eps, zmax_bot+eps};
Physical Surface("bot") = {bot_list()};

//Assign Periodic Boundaries
xmin_per = x_min-buffer_w_size; xmax_per = x_max+buffer_e_size; 
ymin_per = y_min-buffer_s_size; ymax_per = y_max+buffer_n_size; 
per_list1() = Surface In BoundingBox {xmin_per-eps, ymin_per-eps, 0-eps,
                                      xmax_per+eps, ymax_per+eps, 0+eps};
per_list2() = Surface In BoundingBox {xmin_per-eps, ymin_per-eps, Lz-eps,
                                      xmax_per+eps, ymax_per+eps, Lz+eps};
Physical Surface("periodic") = {per_list1(), per_list2()};

//Assign Fluid Volume
volume_list() = Volume In BoundingBox{x_min-buffer_w_size-eps, y_min-buffer_s_size-eps, 0-eps,
                              x_max+buffer_e_size+eps, y_max+buffer_n_size+eps, Lz+eps};
Physical Volume("fluid") = {volume_list()};


//MAKE MESH 
Mesh.MshFileVersion = 2.2;
Mesh 3; //3D


//SET PERIODICITY
// Get all surfaces at zmin (z=0)
Szmin() = Surface In BoundingBox{x_min-buffer_w_size-eps, y_min-buffer_s_size-eps, 0-eps,
                                 x_max+buffer_e_size+eps, y_max+buffer_n_size+eps, 0+eps}; //{xmin,ymin,zmin,xmax,ymax,zmax}

For i In {0:#Szmin()-1}
  // Get the bounding box of each surface at zmin
  bb() = BoundingBox Surface { Szmin(i) };

  // Translate the bounding box to zmax (z=Lz) and look for surfaces inside it:
  Szmax() = Surface In BoundingBox { bb(0)-eps, bb(1)-eps, bb(2)-eps+Lz,
                                     bb(3)+eps, bb(4)+eps, bb(5)+eps+Lz };

  // For all the matches, compare the corresponding bounding boxes...
  For j In {0:#Szmax()-1}
    bb2() = BoundingBox Surface { Szmax(j) };
    bb2(2) -= Lz;
    bb2(5) -= Lz;

    // ...and if they match, apply the periodicity constraint
    If(Fabs(bb2(0)-bb(0)) < eps && Fabs(bb2(1)-bb(1)) < eps &&
       Fabs(bb2(2)-bb(2)) < eps && Fabs(bb2(3)-bb(3)) < eps &&
       Fabs(bb2(4)-bb(4)) < eps && Fabs(bb2(5)-bb(5)) < eps)
      Periodic Surface {Szmax(j)} = {Szmin(i)} Translate {0,0,Lz};
    EndIf
  EndFor
EndFor


EndIf