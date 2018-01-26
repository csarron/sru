#ifndef _SRU_KERNEL
#define _SRU_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void sru_fwd_cu(float* u, float* x, float* bias,
                float* init, float* mask_h, const int len,
                const int batch, const int d, const int k,
                float* h, float* c, const int activation_type, cudaStream_t stream);

void sru_bwd_cu(float* u, float* x,
                float* bias, float* init,
                float* mask_h, float* c,
                float* grad_h, float* grad_last,
                const int len, const int batch, const int d, const int k,
                float* grad_u, float* grad_x, float* grad_bias,
                float* grad_init, int activation_type, cudaStream_t stream);

void sru_bi_fwd_cu(float* u, float* x,
                   float* bias, float* init,
                   float* mask_h, const int len,
                   const int batch, const int d, const int k,
                   float* h, float* c, const int activation_type, cudaStream_t stream);

void sru_bi_bwd_cu(float* u, float* x,
                   float* bias, float* init,
                   float* mask_h, float* c,
                   float* grad_h, float* grad_last,
                   const int len, const int batch, const int d,
                   const int k, float* grad_u, float* grad_x,
                   float* grad_bias, float* grad_init, int activation_type, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
