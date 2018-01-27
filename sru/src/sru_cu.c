#include <THC/THC.h>
#include "sru_kernel.h"

extern THCState *state;

int sru_forward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,THCudaTensor *b_t,
                THCudaTensor *init_t,  THCudaTensor *m_t,  THCudaTensor *h_t, THCudaTensor *c_t,
                int len, int batch, int d, int k, int a_t, int mask_flag)
{
    cudaStream_t stream = THCState_getCurrentStream(state);
    float *u = THCudaTensor_data(state, u_t);
    float *x = THCudaTensor_data(state, x_t);
    float *b = THCudaTensor_data(state, b_t);
    float *init = THCudaTensor_data(state, init_t);

    float *h = THCudaTensor_data(state, h_t);
    float *c = THCudaTensor_data(state, c_t);
    if(k != 3)
    {
        x = NULL;
    }

    float * mask = NULL;
    if(mask_flag > 0)
    {
        mask = THCudaTensor_data(state, m_t);
    }
    sru_fwd_cu(u, x, b, init, mask, len, batch, d, k, h, c, a_t, stream);

    return 1;
}

int sru_backward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,
                THCudaTensor *b_t, THCudaTensor *init_t,
                THCudaTensor *m_t, THCudaTensor *c_t,
                THCudaTensor *gh_t, THCudaTensor *gl_t,
                THCudaTensor *gu_t,  THCudaTensor *gx_t,
                THCudaTensor *gb_t,  THCudaTensor *gi_t,
                int len, int batch, int d, int k,int a_t, int mask_flag)
{
    cudaStream_t stream = THCState_getCurrentStream(state);
    float *u = THCudaTensor_data(state, u_t);
    float *x = THCudaTensor_data(state, x_t);
    float *b = THCudaTensor_data(state, b_t);
    float *init = THCudaTensor_data(state, init_t);

    float *c = THCudaTensor_data(state, c_t);

    float *gh = THCudaTensor_data(state, gh_t);
    float *gl = THCudaTensor_data(state, gl_t);

    float *gu = THCudaTensor_data(state, gu_t);
    float *gx = THCudaTensor_data(state, gx_t);
    float *gb = THCudaTensor_data(state, gb_t);
    float *gi = THCudaTensor_data(state, gi_t);
    if(k != 3)
    {
        x = NULL;
        gx = NULL;
    }

    float * mask = NULL;
    if(mask_flag > 0)
    {
        mask = THCudaTensor_data(state, m_t);
    }
    sru_bwd_cu(u, x, b, init, mask, c, gh, gl, len, batch, d, k, gu, gx, gb, gi, a_t, stream);

    return 1;
}


int sru_bi_forward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,THCudaTensor *b_t,
                THCudaTensor *init_t,  THCudaTensor *m_t,  THCudaTensor *h_t, THCudaTensor *c_t,
                int len, int batch, int d, int k, int a_t, int mask_flag)
{
    cudaStream_t stream = THCState_getCurrentStream(state);
    float *u = THCudaTensor_data(state, u_t);
    float *x = THCudaTensor_data(state, x_t);
    float *b = THCudaTensor_data(state, b_t);
    float *init = THCudaTensor_data(state, init_t);

    float *h = THCudaTensor_data(state, h_t);
    float *c = THCudaTensor_data(state, c_t);
    if(k != 3)
    {
        x = NULL;
    }

    float * mask = NULL;
    if(mask_flag > 0)
    {
        mask = THCudaTensor_data(state, m_t);
    }
    sru_bi_fwd_cu(u, x, b, init, mask, len, batch, d, k, h, c, a_t, stream);

    return 1;
}

int sru_bi_backward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,
                THCudaTensor *b_t, THCudaTensor *init_t,
                THCudaTensor *m_t, THCudaTensor *c_t,
                THCudaTensor *gh_t, THCudaTensor *gl_t,
                THCudaTensor *gu_t,  THCudaTensor *gx_t,
                THCudaTensor *gb_t,  THCudaTensor *gi_t,
                int len, int batch, int d, int k,int a_t, int mask_flag)
{
    cudaStream_t stream = THCState_getCurrentStream(state);
    float *u = THCudaTensor_data(state, u_t);
    float *x = THCudaTensor_data(state, x_t);
    float *b = THCudaTensor_data(state, b_t);
    float *init = THCudaTensor_data(state, init_t);

    float *c = THCudaTensor_data(state, c_t);

    float *gh = THCudaTensor_data(state, gh_t);
    float *gl = THCudaTensor_data(state, gl_t);

    float *gu = THCudaTensor_data(state, gu_t);
    float *gx = THCudaTensor_data(state, gx_t);
    float *gb = THCudaTensor_data(state, gb_t);
    float *gi = THCudaTensor_data(state, gi_t);
    if(k != 3)
    {
        x = NULL;
        gx = NULL;
    }

    float * mask = NULL;
    if(mask_flag > 0)
    {
        mask = THCudaTensor_data(state, m_t);
    }
    sru_bi_bwd_cu(u, x, b, init, mask, c, gh, gl, len, batch, d, k, gu, gx, gb, gi, a_t, stream);

    return 1;
}
