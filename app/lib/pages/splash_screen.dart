import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:googleapis/calendar/v3.dart' as calendar;
import 'package:googleapis/drive/v3.dart' as drive;
import 'package:googleapis/gmail/v1.dart' as gmail;
import 'package:googleapis/sheets/v4.dart' as sheets;
import 'package:googleapis/tasks/v1.dart' as tasks;
import 'package:shared_preferences/shared_preferences.dart';

import 'package:app/helper/ble.dart';
import 'package:app/helper/helper.dart';
import 'package:app/helper/wifi.dart';
import 'package:app/main.dart';
import 'package:app/pages/device.dart';
import 'package:app/pages/sign_in.dart';

GoogleSignInAccount? account;

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _navigateToNext();
  }

  Future<void> _navigateToNext() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      prefs.setBool('blind_support', prefs.getBool('blind_support') ?? false);

      if (!(prefs.getBool('logged') ?? false)) {
        _redirectToSignIn();
        return;
      }

      await _handleGoogleSignIn(prefs);
    } catch (e) {
      debugPrint('Splash navigation error: $e');
      _redirectToSignIn();
    }
  }

  Future<void> _handleGoogleSignIn(SharedPreferences prefs) async {
    final googleSignIn = GoogleSignIn(
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
    );

    account = await googleSignIn.signInSilently();
    if (account == null) {
      _redirectToSignIn();
      return;
    }

    final auth = await account!.authentication;
    final initialData = await get_initial_data(auth);
    authentication_key = initialData[0];
    ble_id = initialData[1];

    await Future.wait([
      scan_devices(),
      is_online(),
    ]);

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => DevicePage(
          user: account!,
          blind_support: prefs.getBool('blind_support') ?? false,
        ),
      ),
    );
  }

  void _redirectToSignIn() {
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const SignInPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Empty container, no UI
    return const SizedBox.shrink();
  }
}