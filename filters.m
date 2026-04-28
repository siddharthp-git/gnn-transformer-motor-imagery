% Parameters
fs = 1000;              % Sampling frequency (Hz)
t = 0:1/fs:1;           % Time vector (1 second duration)
f1 = 50;                % Frequency of sine wave 1 (Hz)
f2 = 200;               % Frequency of sine wave 2 (Hz)
fc_low = 40;           % Cutoff frequency for low-pass filter (Hz)
fc_high = 100;          % Cutoff frequency for high-pass filter (Hz)
order = 4;              % Filter order

% Combine sine waves
signal = sin(2*pi*f1*t) + sin(2*pi*f2*t);

% Design filters
[b_butter_low, a_butter_low] = butter(order, fc_low/(fs/2), 'low');
[b_butter_high, a_butter_high] = butter(order, fc_high/(fs/2), 'high');

[b_cheby1_low, a_cheby1_low] = cheby1(order, 0.5, fc_low/(fs/2), 'low');
[b_cheby1_high, a_cheby1_high] = cheby1(order, 0.5, fc_high/(fs/2), 'high');

[b_bessel_low, a_bessel_low] = besself(order, fc_low/(fs/2));
[b_bessel_high, a_bessel_high] = besself(order, fc_high/(fs/2));

[b_ellip_low, a_ellip_low] = ellip(order, 0.5, 20, fc_low/(fs/2), 'low');
[b_ellip_high, a_ellip_high] = ellip(order, 0.5, 20, fc_high/(fs/2), 'high');

% Filter signals
output_butter_low = filter(b_butter_low, a_butter_low, signal);
output_butter_high = filter(b_butter_high, a_butter_high, signal);

output_cheby1_low = filter(b_cheby1_low, a_cheby1_low, signal);
output_cheby1_high = filter(b_cheby1_high, a_cheby1_high, signal);

output_bessel_low = filter(b_bessel_low, a_bessel_low, signal);
output_bessel_high = filter(b_bessel_high, a_bessel_high, signal);

output_ellip_low = filter(b_ellip_low, a_ellip_low, signal);
output_ellip_high = filter(b_ellip_high, a_ellip_high, signal);

% Plot signals
figure;
subplot(3,2,1); plot(t, signal); title('Original Signal'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,2); plot(t, output_butter_low); title('Butterworth Low-pass'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,3); plot(t, output_cheby1_low); title('Chebyshev Low-pass'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,4); plot(t, output_bessel_low); title('Bessel Low-pass'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,5); plot(t, output_ellip_low); title('Elliptic Low-pass'); xlabel('Time (s)'); ylabel('Amplitude');

figure;
subplot(3,2,1); plot(t, signal); title('Original Signal'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,2); plot(t, output_butter_high); title('Butterworth High-pass'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,3); plot(t, output_cheby1_high); title('Chebyshev High-pass'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,4); plot(t, output_bessel_high); title('Bessel High-pass'); xlabel('Time (s)'); ylabel('Amplitude');
subplot(3,2,5); plot(t, output_ellip_high); title('Elliptic High-pass'); xlabel('Time (s)'); ylabel('Amplitude');

% Frequency response plots
figure;
freqz(b_butter_low, a_butter_low); title('Butterworth Low-pass Frequency Response');
figure;
freqz(b_cheby1_low, a_cheby1_low); title('Chebyshev Low-pass Frequency Response');
figure;
freqz(b_bessel_low, a_bessel_low); title('Bessel Low-pass Frequency Response');
figure;
freqz(b_ellip_low, a_ellip_low); title('Elliptic Low-pass Frequency Response');

figure;
freqz(b_butter_high, a_butter_high); title('Butterworth High-pass Frequency Response');
figure;
freqz(b_cheby1_high, a_cheby1_high); title('Chebyshev High-pass Frequency Response');
figure;
freqz(b_bessel_high, a_bessel_high); title('Bessel High-pass Frequency Response');
figure;
freqz(b_ellip_high, a_ellip_high); title('Elliptic High-pass Frequency Response');
