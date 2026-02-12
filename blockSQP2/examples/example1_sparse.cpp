/*
 * blockSQP -- Sequential quadratic programming for problems with
 *             block-diagonal Hessian matrix.
 * Copyright (C) 2012-2015 by Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>
 *
 * Licensed under the zlib license. See LICENSE for more details.
 */

#include <blockSQP2.hpp>
#include <limits>

namespace blockSQP
{

static double const myInf = std::numeric_limits<double>::infinity();    ///< Used to mark sparse zeros in Jacobian

/**
 * \brief Example implementation of a derived class of the ProblemSpec base class.
 * \author Dennis Janka
 * \date 2012-2015
 * 
 * \modifications
 *  \author Reinhold Wittmann
 *  \date 2023-2025
 */

class MyProblem : public Problemspec{
    public:
        Matrix xi0;                         ///< starting values for the optimization (dim nVar)

    public:
        /// Constructor
        MyProblem( int nVar_,               ///< number of variables
                   int nCon_,               ///< number of constraints
                   int nBlocks_,            ///< number of diagonal blocks in the Hessian
                   int *BlockIdx_,          ///< partition of the variable vector according to diagonal blocks (dim nBlocks+1)
                   const Matrix &lbv,       ///< lower bounds for variables (dim nVar)
                   const Matrix &ubv,       ///< upper bounds for variables (dim nVar)
                   const Matrix &lbc,       ///< lower bounds for constraints (dim nCon)
                   const Matrix &ubc,       ///< upper bounds for constraints (dim nCon)
                   const Matrix &xi0_       ///< starting values for the optimization (dim nVar)
                   );

        ~MyProblem();

        /// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (dense version).
        virtual void initialize( Matrix &xi,            ///< optimization variables
                                 Matrix &lambda,        ///< Lagrange multipliers
                                 Matrix &constrJac      ///< constraint Jacobian (dense)
                                 );

        /// Set initial values for xi (and possibly lambda) and parts of the Jacobian that correspond to linear constraints (sparse version).
        virtual void initialize( Matrix &xi,            ///< optimization variables
                                 Matrix &lambda,        ///< Lagrange multipliers
                                 double *jacNz,        ///< nonzero elements of constraint Jacobian
                                 int *jacIndRow,       ///< row indices of nonzero elements
                                 int *jacIndCol        ///< starting indices of columns
                                 );

        /// Evaluate objective, constraints, and derivatives (dense version).
        virtual void evaluate( const Matrix &xi,        ///< optimization variables
                               const Matrix &lambda,    ///< Lagrange multipliers
                               double *objval,          ///< objective function value
                               Matrix &constr,          ///< constraint function values
                               Matrix &gradObj,         ///< gradient of objective
                               Matrix &constrJac,       ///< constraint Jacobian (dense)
                               SymMatrix *hess,        ///< Hessian of the Lagrangian (blockwise)
                               int dmode,               ///< derivative mode
                               int *info                ///< error flag
                               );

        /// Evaluate objective, constraints, and derivatives (sparse version).
        virtual void evaluate( const Matrix &xi,        ///< optimization variables
                               const Matrix &lambda,    ///< Lagrange multipliers
                               double *objval,          ///< objective function value
                               Matrix &constr,          ///< constraint function values
                               Matrix &gradObj,         ///< gradient of objective
                               double *jacNz,          ///< nonzero elements of constraint Jacobian
                               int *jacIndRow,         ///< row indices of nonzero elements
                               int *jacIndCol,         ///< starting indices of columns
                               SymMatrix *hess,        ///< Hessian of the Lagrangian (blockwise)
                               int dmode,               ///< derivative mode
                               int *info                ///< error flag
                               );

        /// Generic method to convert dense constraint Jacobian to a sparse matrix in Harwell--Boeing (column compressed) format.
        virtual void convertJacobian( const Matrix &constrJac,  ///< constraint Jacobian (dense)
                                      double *jacNz,           ///< nonzero elements of constraint Jacobian
                                      int *jacIndRow,          ///< row indices of nonzero elements
                                      int *jacIndCol,          ///< starting indices of columns
                                      bool firstCall = 0        ///< indicates if this method is called for the first time
                                      );
};


MyProblem::MyProblem( int nVar_, int nCon_, int nBlocks_, int *blockIdx_, const Matrix &lbv, const Matrix &ubv, const Matrix &lbc, const Matrix &ubc, const Matrix &xi0_ ){
    
    nVar = nVar_;
    nCon = nCon_;

    nBlocks = nBlocks_;
    blockIdx = new int[nBlocks+1];
    if( nBlocks == 1 )
    {
        blockIdx[0] = 0;
        blockIdx[1] = nVar;
    }
    else
    {
        for( int i=0; i<nBlocks+1; i++ )
            blockIdx[i] = blockIdx_[i];
    }

    lb_var = lbv;
    ub_var = ubv;
    lb_con = lbc;
    ub_con = ubc;

    objLo = -myInf;
    objUp = myInf;

    xi0.Dimension( nVar );
    for( int i=0; i<nVar; i++ )
        xi0( i ) = xi0_( i );
}

MyProblem::~MyProblem(){
    delete[] blockIdx;
}


void MyProblem::convertJacobian( const Matrix &constrJac, double *jacNz, int *jacIndRow, int *jacIndCol, bool firstCall ){
    int nnz, count, i, j;

    if( firstCall )
    {
        // 1st run: count nonzeros
        nnz = 0;
        for( j=0; j<nVar; j++ )
            for( i=0; i<nCon; i++ )
                if( fabs( constrJac( i, j ) < myInf ) )
                    nnz++;
    }
    else
    {
        nnz = jacIndCol[nVar];
        /* arrays jacInd* are already allocated! */
    }

    // 2nd run: store matrix entries columnwise in jacNz
    count = 0;
    for( j=0; j<nVar; j++ )
    {
        jacIndCol[j] = count;
        for( i=0; i<nCon; i++ )
            if( fabs( constrJac( i, j ) < myInf ) )
            {
                jacNz[count] = constrJac( i, j );
                jacIndRow[count] = i;
                count++;
            }
    }
    jacIndCol[nVar] = count;
    if( count != nnz )
         printf( "Error in convertJacobian: %i elements processed, should be %i elements!\n", count, nnz );
}


void MyProblem::initialize( Matrix &xi, Matrix &lambda, Matrix &constrJac )
{   
    // set initial values for xi and lambda
    lambda.Initialize( 0.0 );
    for( int i=0; i<nVar; i++ )
        xi( i ) = xi0( i );
}


void MyProblem::initialize( Matrix &xi, Matrix &lambda, double *jacNz, int *jacIndRow, int *jacIndCol )
{   
    Matrix constrDummy, gradObjDummy, constrJac;
    SymMatrix *hessDummy = nullptr;
    double objvalDummy;
    int info;

    // set initial values for xi and lambda
    lambda.Initialize( 0.0 );
    for( int i=0; i<nVar; i++ )
        xi( i ) = xi0( i );

    // find out Jacobian sparsity pattern by evaluating derivatives once
    constrDummy.Dimension( nCon ).Initialize( 0.0 );
    gradObjDummy.Dimension( nVar ).Initialize( 0.0 );
    constrJac.Dimension( nCon, nVar ).Initialize( myInf );
    evaluate( xi, lambda, &objvalDummy, constrDummy, gradObjDummy, constrJac, hessDummy, 1, &info );

    // allocate sparse Jacobian structures
    convertJacobian( constrJac, jacNz, jacIndRow, jacIndCol, 0 );
}

/*
 * PROBLEM-SPECIFIC PART STARTS HERE
 */

void MyProblem::evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr,
                          Matrix &gradObj, double *jacNz, int *jacIndRow, int *jacIndCol,
                          SymMatrix *hess, int dmode, int *info){
    Matrix constrJac;
    constrJac.Dimension(nCon, nVar).Initialize(myInf);
    evaluate(xi, lambda, objval, constr, gradObj, constrJac, hess, dmode, info);

    // Convert to sparse format
    if (dmode != 0)
        convertJacobian(constrJac, jacNz, jacIndRow, jacIndCol, 0);
}


void MyProblem::evaluate(const Matrix &xi, const Matrix &lambda, double *objval, Matrix &constr,
                         Matrix &gradObj, Matrix &constrJac, SymMatrix *hess,
                         int dmode, int *info){
    *info = 0;

    /*
     * min   x1**2 - 0.5*x2**2
     * s.t.  x1 - x2 = 0
     */
    if (dmode >= 0){
        *objval = xi(0)*xi(0) - 0.5*xi(1)*xi(1);
        constr(0) = xi(0) - xi(1);
    }

    if (dmode > 0){
        gradObj(0) = 2.0 * xi(0);
        gradObj(1) = -xi(1);

        constrJac(0, 0) = 1.0;
        constrJac(0, 1) = -1.0;
    }
}

} // namespace blockSQP

int main(int argc, const char* argv[]){
    using namespace blockSQP;
    SQPresults ret;
    MyProblem *prob;
    SQPmethod *meth;
    SQPoptions *opts;
    SQPstats *stats;
    char outpath[255];
    strcpy(outpath, "./");

    /*--------------------*/
    /* Setup problem data */
    /*--------------------*/
    int nVar = 2;
    int nCon = 1;

    int nBlocks = nVar;
    int blockIdx[2+1]; //[nBlocks+1]

    // Initial values
    Matrix x0(nVar);
    x0(0) = 10.0;
    x0(1) = 10.0;

    // Variable bounds
    Matrix lb_var, ub_var, lb_con, ub_con;
    lb_var.Dimension(nVar).Initialize(-myInf);
    ub_var.Dimension(nVar).Initialize(myInf);

    // Constraint bounds
    lb_con.Dimension(nCon);
    ub_con.Dimension(nCon);
    lb_con(0) = 0.0;
    ub_con(0) = 0.0;
    
    // Variable partition for block Hessian
    for( int i=0; i<nBlocks+1; i++ )
        blockIdx[i] = i;
    
    // Create problem evaluation object
    prob = new MyProblem( nVar, nCon, nBlocks, blockIdx, lb_var, ub_var, lb_con, ub_con, x0 );
	prob->nnz = 2;
    
    /*------------------------*/
    /* Options for SQP solver */
    /*------------------------*/
    opts = new SQPoptions();
    opts->opt_tol = 1.0e-12;         // For this example, set tolerances very low. The default is 1e-6
    opts->feas_tol = 1.0e-12;        
    
    opts->sparse = true;
    
    opts->enable_linesearch = false;        // Disable filter line search for this example
    opts->hess_approx = Hessians::SR1;           
    opts->fallback_approx = Hessians::BFGS;       // ' ', not needed if hess_approximation is positive definite
    
    opts->sizing = Sizings::None;              // Turn of sizing strategy for this example (1: OL sizing, 2: shanno-phua, 3: geom. mean of 1 and 2, 4: COL sizing)
    opts->fallback_sizing = Sizings::None;     // ' '
    opts->sparse = true;                  // Dense matrices for this example
    opts->print_level = 2;                  // Maximum print output
    opts->debug_level = 0;                  // No printing to files
    
    opts->par_QPs = false;
    opts->max_conv_QPs = 1;
    
    opts->indef_delay = 1;
    

    opts->qpsol = QPsolvers::qpOASES;       // Set QP solver
    qpOASES_options QPopts;                 // Options to be passed to qpOASES
    QPopts.sparsityLevel = 2;               // Select the method qpOASES uses. 0: dense, 1: sparse, 2: schur, requires sparse linear solver such as MUMPS
                                               // Default -1 (automatically infer and SET from SQPoptions). Internal default options may also be overwritten by SQPoptions
    QPopts.printLevel = 0;                  // QP solver options keep their name. See qpOASES manual for options. Currently only printLevel and terminationTolerance can be passed.


    /*-------------------------------------------------*/
    /* Create blockSQP method object and run algorithm */
    /*-------------------------------------------------*/
    stats = new SQPstats( outpath );
    meth = new SQPmethod( prob, opts, stats );

    meth->init();
    std::cout << meth->vars->xi;

    ret = meth->run( 100 );

    meth->finish();
    if ( ret == SQPresults::it_finished)
        printf("\033[0;36m***Maximum number of iterations reached.***\n\033[0m");

    printf("\nPrimal solution:\n");
    meth->vars->xi.Print();
    printf("\nDual solution:\n");
    meth->vars->lambda.Print();
    printf("\nHessian approximation at the solution:\n");
    for( int i=0; i<meth->vars->nBlocks; i++ )
        meth->vars->hess[i].Print();
    
    // Clean up
    delete prob;
    delete stats;
    delete opts;
    delete meth;
}

