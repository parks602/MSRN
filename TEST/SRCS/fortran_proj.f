!*************************************************************************
!  Purpose :
!  Initialization of map projection(LC)
!                                   modified by Jun-Tae Choi Jan. 31, 2005
!*************************************************************************
      SUBROUTINE LAMCINIT(PI_,R_,SLAT1_,SLAT2_,OLAT_,OLON_,XO_,YO_,DD_)
!-------------------------------------------------------------------------
      IMPLICIT NONE
!-------------------------------------------------------------------------
! Declarations of Subroutine Argument
      real  :: PI_,R_,SLAT1_,SLAT2_,OLAT_,OLON_,DD_, XO_, YO_
      real  :: PI,R,SLAT1,SLAT2,SN,SF,RO,OLAT,OLON, XO, YO
!-------------------------------------------------------------------------
! Variables Declaration
      real  :: DEGRAD, RADDEG
      COMMON /LAMCON/ R,SLAT1,SLAT2,SN,SF,RO,OLAT,OLON,XO,YO
      COMMON /CALCUL/ PI,DEGRAD,RADDEG
!f2py real,intent(in) :: PI, R, SLAT1, SLAT2, OLAT, OLON, DD, XO, YO
!-------------------------------------------------------------------------
        XO = XO_
        YO = YO_
        PI = PI_
        DEGRAD = PI / 180.
        RADDEG = 180. / PI
        R = R_/DD_

        SLAT1 = SLAT1_*DEGRAD
        SLAT2 = SLAT2_*DEGRAD
        OLAT = OLAT_*DEGRAD
        OLON = OLON_*DEGRAD


        IF ((SLAT1 == SLAT2).OR. (ABS(SLAT1) >= PI*0.5)  
     +     .OR.(ABS(SLAT2) >= (PI*0.5)))               THEN
           PRINT *,'ERROR  [ LAMCPROJ ]'
           PRINT *,'(SLAT1,SLAT2) :',SLAT1*RADDEG,SLAT2*RADDEG
           STOP
        ENDIF

        SN = TAN(PI*0.25+SLAT2*0.5)/TAN(PI*0.25+SLAT1*0.5)
        SN = ALOG(COS(SLAT1)/COS(SLAT2))/ALOG(SN)
        SF = (TAN(PI*0.25+SLAT1*0.5))**SN*COS(SLAT1)/SN

        IF (ABS(OLAT) > (89.9*DEGRAD)) THEN
           IF (SN*OLAT < 0.0) THEN
              PRINT *,'ERROR  [ LAMCPROJ ]'
              PRINT *,'(SLAT1,SLAT2) :',SLAT1*RADDEG,SLAT2*RADDEG
              PRINT *,'(OLAT ,OLON ) :',OLAT*RADDEG,OLON*RADDEG
              STOP
           ENDIF
           RO = 0.
        ELSE
           RO = R*SF/(TAN(PI*0.25+OLAT*0.5))**SN
        ENDIF
      END SUBROUTINE LAMCINIT
!*************************************************************************
!  Purpose :
!
!  Return lat. longitude from grid value or grid value from lat. longitude 
!  in Lambert Conformal Conic Projection
!
!  ALAT, ALON : (latitude,longitude) at earth  [degree]
!  X, Y       : (x,y) coordinate in map  [grid]
!  * N = 0   : (lat,lon) --> (x,y)
!  * N = 1   : (x,y) --> (lat,lon)
!                                   modified by Jun-Tae Choi Jan. 31, 2005
!*************************************************************************
!  SUBROUTINE LAMCPROJ(ALAT,ALON,X,Y,N,IERR,                              &
!                      PI,R,SLAT1,SLAT2,SN,SF,RO,OLAT,OLON,XO,YO)  
      SUBROUTINE LAMCPROJ(X,Y,LAT,LON)
!-------------------------------------------------------------------------
      IMPLICIT NONE
!-------------------------------------------------------------------------
! Declarations of Subroutine Argument
      real    :: LAT,LON,X,Y
      real    :: R,PI,SN,SF,RO,XO,YO,OLAT,OLON
      real    :: SLAT1,SLAT2
!f2py real,intent(in) :: X,Y

!-------------------------------------------------------------------------
! Variables Declaration

      real    :: DEGRAD, RADDEG
      real    :: RA, THETA, XN, YN
!f2py real, intent(out) :: lat, lon
      COMMON /LAMCON/ R,SLAT1,SLAT2,SN,SF,RO,OLAT,OLON,XO,YO
      COMMON /CALCUL/ PI,DEGRAD,RADDEG
!-------------------------------------------------------------------------
        XN = X - XO
        YN = RO - Y + YO
        RA = SQRT(XN*XN+YN*YN)
        IF (SN.LT.0) RA = -RA

        LAT = (R*SF/RA)**(1./SN)
        LAT = 2.*ATAN(LAT)-PI*0.5
        IF (ABS(XN).LE.0.0) THEN
           THETA = 0.
        ELSE
           IF (ABS(YN).LE.0.0) THEN
              THETA = PI*0.5
              IF (XN.LT.0.0) THETA = -THETA
           ELSE
              THETA = ATAN2(XN,YN)
           ENDIF
        ENDIF

        LON = THETA/SN + OLON
        LAT = LAT*RADDEG
        LON = LON*RADDEG
      END SUBROUTINE LAMCPROJ
