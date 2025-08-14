import 'dart:async';
import 'package:app/helper/socket.dart';
import 'package:app/helper/loading_screen.dart';
import 'package:app/helper/query.dart';
import 'package:app/main.dart';
import 'package:app/pages/device.dart';
import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:googleapis/calendar/v3.dart' as calendar;
import 'package:googleapis/gmail/v1.dart' as gmail;
import 'package:googleapis/drive/v3.dart' as drive;
import 'package:googleapis/tasks/v1.dart' as tasks;
import 'package:googleapis/sheets/v4.dart' as sheets;
import 'package:http/http.dart';
import 'package:http/io_client.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_client/web_socket_client.dart';
import 'dart:math';

// Futuristic AI Color Palette
const _kAiRed = Color(0xFFD10047);  // Roman Red
const _kAiWhite = Color(0xFFFFFFFF);
const _kAiBlack = Color(0xFF000000);
const _kAiDarkGray = Color(0xFF121212);
const _kAiNeonPulse = Color(0x55FF0055);  // Pulsing neon effect

// Global Strings
String authentication_key = '';
String ble_id = '';

class SignInPage extends StatefulWidget {
  const SignInPage({super.key});

  @override
  State<SignInPage> createState() => _SignInPageState();
}

class _SignInPageState extends State<SignInPage> with SingleTickerProviderStateMixin {
  bool isLoading = false;
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  final GoogleSignIn _googleSignIn = GoogleSignIn(
      clientId: CLIENT_ID,
      scopes: [
        calendar.CalendarApi.calendarScope,
        gmail.GmailApi.gmailReadonlyScope,
        gmail.GmailApi.gmailSendScope,
        gmail.GmailApi.gmailComposeScope,
        gmail.GmailApi.gmailModifyScope,
        drive.DriveApi.driveScope,
        tasks.TasksApi.tasksScope,
        sheets.SheetsApi.spreadsheetsScope,
      ],
      forceCodeForRefreshToken: true,
      serverClientId: SERVER_CLIENT_ID);

  GoogleSignInAccount? user;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    
    _pulseAnimation = Tween<double>(begin: 0.9, end: 1.1).animate(
      CurvedAnimation(
        parent: _pulseController,
        curve: Curves.easeInOut,
      ),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  /// Login to Google and then verify account
  Future<void> _login() async {
    try {
      setState(() => isLoading = true);
      final GoogleSignInAccount? account = await _googleSignIn.signIn();
      if (account == null) {
        setState(() => isLoading = false);
        return; // User canceled the sign-in
      }
      final GoogleSignInAuthentication auth = await account.authentication;

      user = account;

      await _verifyAuthentication(auth.idToken, account);
    } catch (error) {
      setState(() => isLoading = false);
      _showDialog('Login Failed', error.toString());
    }
  }

  /// Verify Google Account
  /// Check whether user bough the glasses or not
  ///
  /// Parameters
  ///   - String auth code for verification
  ///   - GoogleSignInAccount user account he signed in with
  Future<void> _verifyAuthentication(
      String? authCode, GoogleSignInAccount? account) async {
    final prefs = await SharedPreferences.getInstance();
    final completer = Completer<String>();
    await socket.connection.firstWhere((state) => state is Connected);

    socket.send('authentication¬$authCode'); // js ws request w authcode

    final subscription = socket.messages.listen((response) {
      completer.complete(response);
    });

    final result = await completer.future;
    await subscription.cancel();

    authentication_key = result;

    if (authentication_key.isEmpty) {
      // No authentication key was sent back -> user didnt buy glasses, no login
      await _googleSignIn.signOut();
      setState(() => isLoading = false);
      _showDialog('Authentication failed',
          'Please log in with an account that has purchased the Gemini Sight Glasses.');
      return;
    }

    // Authentication was successful
    await _handleServerAuthCode(account!.serverAuthCode!);
    await _handleFirstTimeLogin(account, prefs);

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => DevicePage(
          user: user!,
          blind_support: prefs.getBool('blind_support') ?? false,
        ),
      ),
    );
  }

  /// Request to js ws auth code
  Future<void> _handleServerAuthCode(String serverAuthCode) async {
    final completer = Completer<String>();
    await socket.connection.firstWhere((state) => state is Connected);

    socket.send('auth_code¬$authentication_key¬$serverAuthCode');

    final subscription = socket.messages.listen((response) {
      completer.complete(response);
    });

    await completer.future;
    await subscription.cancel();
  }

  /// Handle first time login for user
  /// Start query to learn on data
  /// Change first_time in db
  Future<void> _handleFirstTimeLogin(
      GoogleSignInAccount? account, SharedPreferences prefs) async {
    final completer = Completer<String>();
    await socket.connection.firstWhere((state) => state is Connected);

    socket.send('first_time¬$authentication_key¬${user!.email}');

    final subscription = socket.messages.listen((response) {
      completer.complete(response);
    });

    final result = await completer.future;
    await subscription.cancel();

    if (result == "true") {
      await get_query(account!, context);
    }

    await prefs.setBool('logged', true);
    await prefs.setBool('first_time', false);

    socket.send('not_first_time¬$authentication_key');
  }

  void _showDialog(String title, String content) {
    showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: _kAiDarkGray,
          title: Text(title, style: const TextStyle(color: _kAiWhite)),
          content: SingleChildScrollView(
            child: ListBody(
              children: <Widget>[
                Text(content, style: const TextStyle(color: _kAiWhite)),
              ],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Okay', style: TextStyle(color: _kAiRed)),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _kAiBlack,
      body: Stack(
        children: [
          // Background elements
          Positioned.fill(
            child: CustomPaint(
              painter: _GridPatternPainter(),
            ),
          ),
          
          // Floating particles
          ..._buildFloatingParticles(),
          
          // Main content
          Center(
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // App logo
                  AnimatedBuilder(
                    animation: _pulseAnimation,
                    builder: (context, _) {
                      return Transform.scale(
                        scale: _pulseAnimation.value,
                        child: Container(
                          width: 140,
                          height: 140,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _kAiDarkGray,
                            border: Border.all(
                              color: _kAiRed,
                              width: 2.0,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: _kAiRed.withOpacity(0.3),
                                blurRadius: 20,
                                spreadRadius: 5,
                              ),
                            ],
                          ),
                          child: Center(
                            child: Image.asset(
                              'assets/images/logo.png',
                              width: 60,
                              height: 60,
                              fit: BoxFit.contain,
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                  
                  const SizedBox(height: 40),
                  
                  // App title
                  const Text(
                    'TACIT',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      color: _kAiWhite,
                      letterSpacing: 4.0,
                    ),
                  ),
                  
                  const SizedBox(height: 8),
                  
                  // Short hero description
                    Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 25.0),
                    child: Text(
                      'Tacit lets you “speak” silently using a lightweight sEMG headset. It reads subtle muscle signals and turns them into speech, no overhearing. Private, hands-free, and assistive.',
                      style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w300,
                      color: _kAiWhite.withOpacity(0.7),
                      letterSpacing: 1.2,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    ),
                  
                  const SizedBox(height: 40),
                  
                  // Login button
                  if (!isLoading) ...[
                    AnimatedBuilder(
                      animation: _pulseAnimation,
                      builder: (context, _) {
                        return Transform.scale(
                          scale: _pulseAnimation.value,
                          child: ElevatedButton(
                            onPressed: _login,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              foregroundColor: _kAiWhite,
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 32, vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                                side: BorderSide(color: _kAiRed, width: 2),
                              ),
                              shadowColor: _kAiRed.withOpacity(0.4),
                              elevation: 8,
                            ),
                            child: Container(
                              constraints: const BoxConstraints(minWidth: 200),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Image.asset(
                                    'assets/images/google_logo.png', 
                                    height: 24,
                                    width: 24,
                                  ),
                                  const SizedBox(width: 16),
                                  const Text(
                                    'SIGN IN WITH GOOGLE',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: 1.2,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ] else ...[
                    // Loading indicator
                    Column(
                      children: [
                        Container(
                          width: 60,
                          height: 60,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _kAiRed.withOpacity(0.1),
                            border: Border.all(color: _kAiRed, width: 1.5),
                          ),
                          child: const Center(
                            child: CircularProgressIndicator(
                              color: _kAiRed,
                              strokeWidth: 2,
                            ),
                          ),
                        ),
                        const SizedBox(height: 20),
                        const Text(
                          'AUTHENTICATING...',
                          style: TextStyle(
                            color: _kAiRed,
                            fontSize: 14,
                            letterSpacing: 2.0,
                          ),
                        ),
                      ],
                    ),
                  ],
                  
                  const SizedBox(height: 40),
                  
                  // Copyright at the bottom
                  Padding(
                    padding: const EdgeInsets.only(top: 40.0),
                    child: Text(
                      '© 2026 Alex Steiner',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: _kAiWhite.withOpacity(0.4),
                        fontSize: 12,
                        letterSpacing: 0.8,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildFloatingParticles() {
    return [
      Positioned(
        top: 0.15 * MediaQuery.of(context).size.height,
        left: 0.2 * MediaQuery.of(context).size.width,
        child: _FloatingParticle(
          size: 6,
          color: _kAiRed,
        ),
      ),
      Positioned(
        top: 0.4 * MediaQuery.of(context).size.height,
        right: 0.3 * MediaQuery.of(context).size.width,
        child: _FloatingParticle(
          size: 4,
          color: _kAiRed.withOpacity(0.7),
        ),
      ),
      Positioned(
        bottom: 0.3 * MediaQuery.of(context).size.height,
        left: 0.15 * MediaQuery.of(context).size.width,
        child: _FloatingParticle(
          size: 5,
          color: _kAiRed.withOpacity(0.8),
        ),
      ),
    ];
  }
}

class _GridPatternPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = _kAiDarkGray
      ..strokeWidth = 0.8
      ..style = PaintingStyle.stroke;

    // Draw vertical lines
    for (double x = 0; x < size.width; x += 40) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }

    // Draw horizontal lines
    for (double y = 0; y < size.height; y += 40) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _FloatingParticle extends StatefulWidget {
  final double size;
  final Color color;
  
  const _FloatingParticle({
    required this.size,
    required this.color,
  });

  @override
  State<_FloatingParticle> createState() => _FloatingParticleState();
}

class _FloatingParticleState extends State<_FloatingParticle>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _opacity;
  late Animation<Offset> _offset;

  @override
  void initState() {
    super.initState();
    
    final random = Random();
    final offsetX = random.nextDouble() * 20 - 10;
    final offsetY = random.nextDouble() * 20 - 10;
    
    _controller = AnimationController(
      vsync: this,
      duration: Duration(seconds: 2 + random.nextInt(3)),
    )..repeat(reverse: true);
    
    _opacity = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 0.0, end: 1.0), weight: 1),
      TweenSequenceItem(tween: Tween(begin: 1.0, end: 0.0), weight: 1),
    ]).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));
    
    _offset = Tween<Offset>(
      begin: Offset.zero,
      end: Offset(offsetX, offsetY),
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        return Transform.translate(
          offset: _offset.value,
          child: Opacity(
            opacity: _opacity.value,
            child: Container(
              width: widget.size,
              height: widget.size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: widget.color,
                boxShadow: [
                  BoxShadow(
                    color: widget.color,
                    blurRadius: 8,
                    spreadRadius: 2,
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class GoogleAPIClient extends IOClient {
  final Map<String, String> _headers;

  GoogleAPIClient(this._headers) : super();

  @override
  Future<IOStreamedResponse> send(BaseRequest request) =>
      super.send(request..headers.addAll(_headers));

  @override
  Future<Response> head(Uri url, {Map<String, String>? headers}) =>
      super.head(url,
          headers: headers != null ? (headers..addAll(_headers)) : _headers);
}