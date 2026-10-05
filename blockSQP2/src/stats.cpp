/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

/*
 * blockSQP2 -- A structure-exploiting nonlinear programming solver based
 *              on blockSQP by Dennis Janka.
 * Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>
 * 
 * Licensed under the zlib license. See LICENSE for more details.
 */

/**
 * \file blocksqp_stats.cpp
 * \author Dennis Janka
 * \date 2012-2015
 *
 *  Implementation of SQPstats class that records statistics during an SQP run.
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

#include <blockSQP2/iterate.hpp>
#include <blockSQP2/options.hpp>
#include <blockSQP2/stats.hpp>
#include <blockSQP2/general_purpose.hpp>

namespace blockSQP2{

SQPstats::SQPstats(PATHSTR myOutpath){
    strcpy(outpath, myOutpath);
    
    itCount = 0;
    qpItTotal = 0;
    qpIterations = 0;
    qpIterations2 = 0;
    qpResolve = 0;
    rejectedSR1 = 0;
    hessSkipped = 0;
    hessDamped = 0;
    averageSizingFactor = 0.0;
    nFunCalls = 0;
    nDerCalls = 0;
    nRestHeurCalls = 0;
    nRestPhaseCalls = 0;
    
    nTotalUpdates = 0;
    nTotalSkippedUpdates = 0;
}


inline void printHeading(SQPiterate *vars, SQPoptions *param){
    // printf("%-6s", "   it");
    // printf("%-17s", " qpIt");
    // printf("%-9s", "obj");
    // printf("%-11s", "feas");
    // printf("%-6s", "opt");
    
    printf("%5s", "it");
    printf("%6s", "QPit");
    printf("%15s", "obj");
    
    printf("%10s", "feas");
    printf("%10s", "opt");
    
    if (param->print_level > 1){
        // printf("%-11s", "|Lgrad|");
        // printf("%-9s", "|step|");
        // printf("%-11s", "|lstep|");
        
        printf("%10s", "|Lgrad|");
        printf("%10s", "|step|");
        printf("%10s", "|lstep|");
    }
    // printf("%-8s", "alpha");
    printf("%9s", "alpha");
    
    
    // if (param->print_level > 1){
    //     printf("%-6s","nSOCS");
    //     printf("%-18s","sk, da, sca");
    //     printf("%-6s","QPr,mu");
    // }
    
    // printf("%-6s", "QPk");
    printf("%5s", "QPk");
    
    printf("\n");
}

void SQPstats::printProgress(Problemspec *prob, SQPiterate *vars, SQPoptions *param, bool hasConverged){
    if (itCount == 0){
        if (param->print_level > 0){
            prob->printInfo();
            printHeading(vars, param);
            
            // // Values for first iteration
            printf("%5i  ", itCount);
            printf("%5s ", "");
            printf("% 10e  ", vars->obj);
            printf("%-10.2e", vars->cNorm);//cNorm instead of cNormS
            printf("%-10.2e", vars->tol);
            printf("\n");
        }
        
        if (param->debug_level > 0){
            // Print everything in a CSV file as well
            fprintf( progressFile, "%23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %i, %i, %23.16e, %i, %23.16e\n",
                    vars->obj, vars->cNormS, vars->tol, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0, 0.0 );
        }
    }
    else{
        // Every twenty iterations print headline
        if (itCount % 20 == 0 && param->print_level > 0){
            printHeading(vars, param);
        }
        
        // All values
        if( param->print_level > 0){
            printf("%5i ", itCount);
            printf("%5i  ", qpIterations);
            printf("% 10e  ", vars->obj);
            printf("%-10.2e", vars->cNorm);
            printf("%-10.2e", vars->tol);
            if (param->print_level > 1){
                printf("%-10.2e", vars->gradNorm);
                printf("%-10.2e", lInfVectorNorm(vars->deltaXi));
                printf("%-10.2e", vars->lambdaStepNorm);
            }
            
            if (vars->steptype == StepTypes::rest_phase) printf("%-9s", "1.0 (r)");
            else if ((vars->alpha == 1.0 && vars->steptype != StepTypes::KKT_heuristic) || !param->print_colored)
                printf("%-9.1e", vars->alpha);
            else printf("\033[0;36m%-9.1e\033[0m", vars->alpha);
            
            // if (param->print_level > 1){
            //     if (vars->nSOCS == 0 || !param->print_colored)
            //         printf("%5i", vars->nSOCS);
            //     else
            //         printf("\033[0;36m%5i\033[0m", vars->nSOCS);
            //     printf("%3i, %3i, %-9.1e", hessSkipped, hessDamped, averageSizingFactor);
            //     printf("%i", vars->QP_num_accepted);
            // }
            printf("%i", vars->QP_num_accepted);
            printf("\n");
        }

        if (param->debug_level > 0){
            fprintf(progressFile, "%23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %23.16e, %i, %i, %i, %23.16e, %i\n",
                    vars->obj, vars->cNormS, vars->tol, vars->gradNorm, lInfVectorNorm( vars->deltaXi ),
                    vars->lambdaStepNorm, vars->alpha, vars->nSOCS, hessSkipped, hessDamped, averageSizingFactor,
                    qpResolve);
            fprintf( updateFile, "%i\t", qpResolve );
        }
    }
    
    printDebug( vars, param );
    
    hessSkipped = 0;
    hessDamped = 0;
    qpItTotal += qpIterations;
    qpItTotal += qpIterations2;
    qpIterations = 0;
    qpIterations2 = 0;
    qpResolve = 0;
}


void SQPstats::initStats(SQPoptions *param){
    PATHSTR filename;

    // Open files
    if (param->debug_level > 0){
        // SQP progress
        strcpy( filename, outpath );
        strcat( filename, "sqpits.csv" );
        progressFile = fopen( filename, "w");

        // Update sequence
        strcpy( filename, outpath );
        strcat( filename, "updatesequence.txt" );
        updateFile = fopen( filename, "w" );
    }

    if (param->debug_level > 1){
        // Primal variables
        strcpy( filename, outpath );
        strcat( filename, "pv.csv" );
        primalVarsFile = fopen( filename, "w");

        // Dual variables
        strcpy( filename, outpath );
        strcat( filename, "dv.csv" );
        dualVarsFile = fopen( filename, "w");
    }

    itCount = 0;
    qpItTotal = 0;
    qpIterations = 0;
    hessSkipped = 0;
    hessDamped = 0;
    averageSizingFactor = 0.0;
}


void SQPstats::printPrimalVars( const Matrix &xi )
{
    for( int i=0; i<xi.M()-1; i++ )
        fprintf( primalVarsFile, "%23.16e ", xi( i ) );
    fprintf( primalVarsFile, "%23.16e\n", xi( xi.M()-1 ) );
}


void SQPstats::printDualVars( const Matrix &lambda )
{
    for( int i=0; i<lambda.M()-1; i++ )
        fprintf( dualVarsFile, "%23.16e ", lambda( i ) );
    fprintf( dualVarsFile, "%23.16e\n", lambda( lambda.M()-1 ) );
}


void SQPstats::printHessian( int nBlocks, SymMatrix *&hess ){
    PATHSTR filename;
    int offset, i, j, iBlock, nVar;

    nVar = 0;
    for( iBlock=0; iBlock<nBlocks; iBlock++ )
        nVar += hess[iBlock].m;

    SymMatrix fullHessian;
    fullHessian.Dimension( nVar ).Initialize( 0.0 );

    strcpy( filename, outpath );
    strcat( filename, "hes.m" );
    hessFile = fopen( filename, "w");

    offset = 0;
    for (iBlock = 0; iBlock < nBlocks; iBlock++){
        for( i=0; i<hess[iBlock].m; i++ )
            for( j=i; j<hess[iBlock].m; j++ )
                fullHessian( offset + i, offset + j ) = hess[iBlock]( i,j );

        offset += hess[iBlock].m;
    }

    fprintf( hessFile, "H=" );
    fullHessian.Print( hessFile, 23, 1 );
    fprintf( hessFile, "\n" );
    fclose( hessFile );
}


void SQPstats::printHessian( int nVar, double *hesNz, int *hesIndRow, int *hesIndCol )
{
    PATHSTR filename;

    strcpy( filename, outpath );
    strcat( filename, "hes.dat" );
    hessFile = fopen( filename, "w");

    printSparseMatlab( hessFile, nVar, nVar, hesNz, hesIndRow, hesIndCol );

    fprintf( hessFile, "\n" );
    fclose( hessFile );
}


void SQPstats::printJacobian( const Matrix &constrJac )
{
    PATHSTR filename;

    strcpy( filename, outpath );
    strcat( filename, "jac.m" );
    jacFile = fopen( filename, "w");

    fprintf( jacFile, "A=" );
    constrJac.Print( jacFile, 23, 1 );
    fprintf( jacFile, "\n" );

    fclose( jacFile );
}


void SQPstats::printJacobian( int nCon, int nVar, double *jacNz, int *jacIndRow, int *jacIndCol )
{
    PATHSTR filename;

    strcpy( filename, outpath );
    strcat( filename, "jac.dat" );
    jacFile = fopen( filename, "w");

    printSparseMatlab( jacFile, nCon, nVar, jacNz, jacIndRow, jacIndCol );

    fprintf( jacFile, "\n" );
    fclose( jacFile );
}


void SQPstats::printSparseMatlab( FILE *file, int nRow, int nCol, double *nz, int *indRow, int *indCol )
{
    int i, j, count;

    count = 0;
    fprintf( file, "%i %i 0\n", nRow, nCol );
    for( i=0; i<nCol; i++ )
        for( j=indCol[i]; j<indCol[i+1]; j++ )
        {
            // +1 for MATLAB indices!
            fprintf( file, "%i %i %23.16e\n", indRow[count]+1, i+1, nz[count] );
            count++;
        }
}


void SQPstats::printDebug( SQPiterate *vars, SQPoptions *param )
{
    if( param->debug_level > 1 )
    {
        printPrimalVars( vars->xi );
        printDualVars( vars->lambda );
    }
}


void SQPstats::finish( SQPoptions *param )
{
    if( param->debug_level > 0 )
    {
        fprintf( progressFile, "\n" );
        fclose( progressFile );
        fprintf( updateFile, "\n" );
        fclose( updateFile );
    }

    if( param->debug_level > 1 )
    {
        fclose( primalVarsFile );
        fclose( dualVarsFile );
    }
}


void SQPstats::printCppNull( FILE *outfile, char* varname )
{
    fprintf( outfile, "    double *%s = NULL;\n", varname );
}


void SQPstats::printVectorCpp( FILE *outfile, double *vec, int len, char* varname )
{
    int i;

    fprintf( outfile, "    double %s[%i] = { ", varname, len );
    for( i=0; i<len; i++ )
    {
        fprintf( outfile, "%23.16e", vec[i] );
        if( i != len-1 )
            fprintf( outfile, ", " );
        if( (i+1) % 10 == 0 )
            fprintf( outfile, "\n          " );
    }
    fprintf( outfile, " };\n\n" );
}


void SQPstats::printVectorCpp( FILE *outfile, int *vec, int len, char* varname )
{
    int i;

    fprintf( outfile, "    int %s[%i] = { ", varname, len );
    for( i=0; i<len; i++ )
    {
        fprintf( outfile, "%i", vec[i] );
        if( i != len-1 )
            fprintf( outfile, ", " );
        if( (i+1) % 15 == 0 )
            fprintf( outfile, "\n          " );
    }
    fprintf( outfile, " };\n\n" );
}


void SQPstats::dumpQPMatlab( Problemspec *prob, SQPiterate *vars, int sparse_mode )
{
    Matrix temp;
    PATHSTR filename;
    FILE *qpFile;
    FILE *vecFile;

    // Print vectors g, lb, lu, lbA, luA
    strcpy( filename, outpath );
    strcat( filename, "vec.m" );
    vecFile = fopen( filename, "w");

    fprintf( vecFile, "g=" );
    vars->gradObj.Print( vecFile, 23, 1 );
    fprintf( vecFile, "\n\n" );

    temp.Submatrix( vars->delta_lb_var, prob->nVar, 1, 0, 0 );
    fprintf( vecFile, "lb=" );
    temp.Print( vecFile, 23, 1 );
    fprintf( vecFile, "\n\n" );

    temp.Submatrix( vars->delta_ub_var, prob->nVar, 1, 0, 0 );
    fprintf( vecFile, "lu=" );
    temp.Print( vecFile, 23, 1 );
    fprintf( vecFile, "\n\n" );

    temp.Submatrix( vars->delta_lb_con, prob->nCon, 1, 0, 0 );
    fprintf( vecFile, "lbA=" );
    temp.Print( vecFile, 23, 1 );
    fprintf( vecFile, "\n\n" );

    temp.Submatrix( vars->delta_ub_con, prob->nCon, 1, 0, 0 );
    fprintf( vecFile, "luA=" );
    temp.Print( vecFile, 23, 1 );
    fprintf( vecFile, "\n" );

    fclose( vecFile );

    // Print sparse Jacobian and Hessian
    if( sparse_mode )
    {
        printJacobian( prob->nCon, prob->nVar, vars->sparse_constrJac.nz.get(), vars->sparse_constrJac.row.get(), vars->sparse_constrJac.colind.get());
        //printHessian( prob->nVar, vars->hessNz, vars->hessIndRow, vars->hessIndCol );
    }

    // Print a script that correctly reads everything
    strcpy( filename, outpath );
    strcat( filename, "getqp.m" );
    qpFile = fopen( filename, "w");

    fprintf( qpFile, "%% Read vectors g, lb, lu, lbA, luA\n" );
    fprintf( qpFile, "vec;\n" );
    fprintf( qpFile, "%% Read sparse Jacobian\n" );
    fprintf( qpFile, "load jac.dat\n" );
    fprintf( qpFile, "if jac(1) == 0\n" );
    fprintf( qpFile, "    A = [];\n" );
    fprintf( qpFile, "else\n" );
    fprintf( qpFile, "    A = spconvert( jac );\n" );
    fprintf( qpFile, "end\n" );
    fprintf( qpFile, "%% Read sparse Hessian\n" );
    fprintf( qpFile, "load hes.dat\n" );
    fprintf( qpFile, "H = spconvert( hes );\n" );

    fclose( qpFile );
}

} // namespace blockSQP2
