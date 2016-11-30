
import numpy as np

angular_momenta_dict = {"S":    (0,0,0,0),
                        "Px":   (1,0,0,1),
                        "Py":   (0,1,0,1),
                        "Pz":   (0,0,1,1),
                        "Dxx":  (2,0,0,2),
                        "Dxy":  (1,1,0,2),
                        "Dxz":  (1,0,1,2),
                        "Dyy":  (0,2,0,2),
                        "Dyz":  (0,1,1,2),
                        "Dzz":  (0,0,2,2)}

# ==== Functions

def chiRealSpace( Xs, Ys, Zs, cgfs, coefs ):
    R2    = Xs**2 + Ys**2 + Zs**2 
    #invR  = 1/np.sqrt( R2 )
    wf = np.zeros( Xs.shape )
    for i, cgf in enumerate(cgfs):
        cs, es          = cgf.primitives
        label           = cgf.orbType
        chi = np.zeros( Xs.shape )
        #sys.stdout.write( label+" "+str(coefs[i])+" " )
        for c,eta in zip( cs, es ):
            chi += np.exp( -eta*R2 )   # this is the slowest part
        L = angular_momenta_dict[label]
        for j in range(L[0]):
            chi *= Xs             #*invR;  sys.stdout.write("X")
        for j in range(L[1]):
            chi *= Ys             #*invR; sys.stdout.write("Y")
        for j in range(L[2]):
            chi *= Zs             #*invR; sys.stdout.write("Z") 
        #print() 
             
        wf  += chi * coefs[i] 
    return wf
    



XSF_HEAD_DEFAULT = headScan='''
ATOMS
 1   0.0   0.0   0.0

BEGIN_BLOCK_DATAGRID_3D                        
   some_datagrid      
   BEGIN_DATAGRID_3D_whatever 
'''    

def writeArr(f, arr):
    f.write(" ".join(str(x) for x in arr) + "\n")

def writeArr2D(f, arr):
    for vec in arr:
        writeArr(f,vec)
        
def atoms_to_file( f, lvec, symbols, coords ):
    f.write("CRYSTAL\n")
    f.write("PRIMVEC\n")
    writeArr2D(f,lvec)    
    f.write("CONVVEC\n")
    writeArr2D(f,lvec)        
    f.write("PRIMCOORD\n")
    f.write("%i %i\n" %(len(symbols),1) )
    for i in range(len(symbols)):
        f.write("%s %5.6f %5.6f %5.6f\n" %(symbols[i],coords[i,0],coords[i,1],coords[i,2]) )
    
def saveXSF(fname, data, lvec, head=XSF_HEAD_DEFAULT, symbols=None, coords=None ):
    print ( "saving Xsf to ", fname )
    fileout = open(fname, 'w')
    if symbols is not None:
        atoms_to_file( fileout, lvec[1:], symbols, coords )
    fileout.write ("\n")
    fileout.write ("BEGIN_BLOCK_DATAGRID_3D  \n")    
    fileout.write ("   some_datagrid    \n")
    fileout.write ("   BEGIN_DATAGRID_3D_whatever \n")
    nDim = np.shape(data)
    writeArr (fileout, (nDim[2],nDim[1],nDim[0]) )
    writeArr2D(fileout,lvec)
    for r in data.flat:
        fileout.write( "%10.5e\n" % r )
    fileout.write ("   END_DATAGRID_3D\n")
    fileout.write ("END_BLOCK_DATAGRID_3D\n")

def pre_wf_real( coords, Rcut = 6.0, dstep=np.array((0.5,0.5,0.5)) ):
    dstep_inv = 1/dstep
    Rmargin = np.ones(3)*Rcut*1.1
    amin  = np.amin(coords, axis=0) - Rmargin
    amax  = np.amax(coords, axis=0) + Rmargin
    aspan = amax-amin
    nDim  = (aspan*dstep_inv).astype(int)
    ntot  = nDim[0]*nDim[1]*nDim[2]
    print("amin",amin,"amax", amax,"aspan", aspan,"nDim", nDim, "ntot", ntot)
    (Xs,Ys,Zs) = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]]
    Xs = (Xs*dstep[0]) + amin[0]
    Ys = (Ys*dstep[1]) + amin[1]
    Zs = (Zs*dstep[2]) + amin[2]
    Rcuts  = np.ones(coords.shape) * Rcut  # this can be modified later for each atom independently
    minInds= ((coords-Rcuts-amin[None,:] )*dstep_inv[None,:] ).astype(int)
    maxInds= ((coords+Rcuts-amin[None,:] )*dstep_inv[None,:] ).astype(int)
    #print( minInds )
    #print( maxInds )
    XYZs      = (Xs,Ys,Zs)
    indBounds = (minInds, maxInds)
    return XYZs, indBounds, (amin,amax,aspan)

def wf_real( orbital, symbols, coords, dictCGFs, mo_i, XYZs, indBounds ):    

    (Xs,Ys,Zs)         = XYZs
    (minInds, maxInds) = indBounds
        
    wf = np.zeros( Xs.shape )
    imu = 0
    for iatom in range(len(coords)):
        symbol = symbols[iatom]
        print( iatom, symbol )
        ixmin = minInds[iatom,0]; ixmax = maxInds[iatom,0];
        iymin = minInds[iatom,1]; iymax = maxInds[iatom,1];
        izmin = minInds[iatom,2]; izmax = maxInds[iatom,2];
        Xc    = Xs[ixmin:ixmax,iymin:iymax,izmin:izmax]
        Yc    = Ys[ixmin:ixmax,iymin:iymax,izmin:izmax]
        Zc    = Zs[ixmin:ixmax,iymin:iymax,izmin:izmax]
        xyz   = coords[iatom]
        cgfs  = dictCGFs[symbol]
        dim_cgfs  = len(cgfs)
        coefs     = mo_i[imu:imu+dim_cgfs]
        wf[ixmin:ixmax,iymin:iymax,izmin:izmax] += chiRealSpace( Xc-xyz[0], Yc-xyz[1], Zc-xyz[2], cgfs, coefs )
        imu+=dim_cgfs
        #break;    
    return wf
