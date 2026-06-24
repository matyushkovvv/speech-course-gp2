import torch
import torch.nn.functional as F
import torchaudio

from config import TrainConfig


def discriminator_loss(real_outs, fake_outs):
    """LS-GAN потеря дискриминатора."""
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss += torch.mean((r - 1.0) ** 2) + torch.mean(f ** 2)
    return loss


def generator_adversarial_loss(fake_outs):
    """Adversarial потеря генератора (LS-GAN)."""
    loss = 0.0
    for f in fake_outs:
        loss += torch.mean((f - 1.0) ** 2)
    return loss


def feature_matching_loss(real_fmaps, fake_fmaps):
    """L1 между промежуточными картами признаков реального и сгенерированного аудио."""
    loss = 0.0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for rl, fl in zip(real_layers, fake_layers):
            loss += F.l1_loss(fl, rl.detach())
    return loss


def mel_reconstruction_loss(y_real: torch.Tensor, y_fake: torch.Tensor, cfg: TrainConfig):
    """
    L1 между мел-спектрограммами реального и сгенерированного сигнала.
    torch.stft считается на CPU — не поддерживается на MPS.
    """
    orig_device = y_real.device
    window = torch.hann_window(cfg.win_length)
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=cfg.n_fft // 2 + 1,
        f_min=cfg.mel_fmin,
        f_max=cfg.mel_fmax,
        n_mels=cfg.num_mels,
        sample_rate=cfg.sample_rate,
        norm=None,
        mel_scale="htk",
    )

    def to_mel(audio):
        audio_cpu = audio.squeeze(1).cpu()
        stft = torch.stft(
            audio_cpu.reshape(-1, audio_cpu.shape[-1]),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            return_complex=True,
        )
        mag = stft.abs()
        mel = torch.matmul(mag.permute(0, 2, 1), mel_fb).permute(0, 2, 1)
        mel = mel ** cfg.mel_power
        return torch.log(torch.clamp(mel, min=1e-5)).to(orig_device)

    return F.l1_loss(to_mel(y_fake), to_mel(y_real))


def generator_loss(
    y_real, y_fake,
    mpd_real_outs, mpd_fake_outs, mpd_real_fmaps, mpd_fake_fmaps,
    msd_real_outs, msd_fake_outs, msd_real_fmaps, msd_fake_fmaps,
    cfg: TrainConfig,
):
    """Суммарная потеря генератора: adversarial + feature matching + mel-L1."""
    adv = (
        generator_adversarial_loss(mpd_fake_outs)
        + generator_adversarial_loss(msd_fake_outs)
    )
    fm = cfg.lambda_fm * (
        feature_matching_loss(mpd_real_fmaps, mpd_fake_fmaps)
        + feature_matching_loss(msd_real_fmaps, msd_fake_fmaps)
    )
    mel = cfg.lambda_mel * mel_reconstruction_loss(y_real.unsqueeze(1), y_fake, cfg)
    return adv + fm + mel, adv.item(), fm.item(), mel.item()
