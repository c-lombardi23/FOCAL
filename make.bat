@ECHO OFF

REM Get the directory of this script
pushd %~dp0

REM Set up our environment variables
set SPHINXBUILD=sphinx-build
set SPHINXAPIDOC=sphinx-apidoc
set SOURCEDIR=source
set BUILDDIR=build
set PACKAGE_PATH=..\src\cleave_app

REM --- Check if sphinx-build is installed ---
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed and that its script directory is in your PATH.
	echo.
	goto end
)

REM --- Command Dispatching ---
if "%1" == "" goto help
if "%1" == "html" goto html
if "%1" == "clean" goto clean

REM Catch-all for other commands like "latexpdf", etc.
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end


REM --- Custom Rule for 'html' ---
:html
    echo.
    echo ==========================================================
    echo  RUNNING APIDOC TO GENERATE SOURCE (.rst) FILES
    echo ==========================================================
    REM First, clean the old api directory
    IF EXIST %SOURCEDIR%\api (
        rmdir /s /q %SOURCEDIR%\api
    )
    REM Then, run the correct sphinx-apidoc command with the -M flag
    %SPHINXAPIDOC% -f -e -M -o %SOURCEDIR%\api %PACKAGE_PATH%
    
    echo.
    echo ==========================================================
    echo  BUILDING HTML
    echo ==========================================================
    %SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
    
    echo.
    echo  Build finished. Open '%BUILDDIR%\index.html' in your browser.
    echo.
    goto end


REM --- Custom Rule for 'clean' ---
:clean
    echo.
    echo ==========================================================
    echo  CLEANING BUILD AND API DIRECTORIES
    echo ==========================================================
    IF EXIST %BUILDDIR% (
        rmdir /s /q %BUILDDIR%
    )
    IF EXIST %SOURCEDIR%\api (
        rmdir /s /q %SOURCEDIR%\api
    )
    echo  Cleaning complete.
    echo.
    goto end


REM --- Help Rule ---
:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%


:end
popd