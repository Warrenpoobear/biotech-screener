#!/usr/bin/env python3
"""Tests for tools/provision_tenant.py.

The tool exists because there was no supported way to add a tenant to the credstore —
the smoke test had to hand-write a Python snippet. The properties worth pinning:

* secrets are never accepted on the command line (argv is world-readable via ps and
  /proc/<pid>/cmdline, the same reasoning that keeps the bearer off mcp_order_worker's
  argv)
* a typo'd password cannot be silently provisioned — confirmation is required
* re-provisioning an existing tenant is refused unless explicitly requested, so a
  password reset cannot happen by accident
"""

import sqlite3

import pytest

from common.credstore import CredentialStore, generate_key
from tools.provision_tenant import ProvisionError, build_parser, provision


@pytest.fixture()
def store(tmp_path):
    return CredentialStore(tmp_path / "cred.db", key=generate_key())


class TestNoSecretsInArgv:
    def test_parser_has_no_password_option(self):
        opts = {a for act in build_parser()._actions for a in act.option_strings}
        for forbidden in ("--password", "--pass", "--pw"):
            assert forbidden not in opts, forbidden + " must not be accepted on the command line"

    def test_parser_has_no_bearer_option(self):
        opts = {a for act in build_parser()._actions for a in act.option_strings}
        for forbidden in ("--bearer", "--token", "--robinhood-bearer"):
            assert forbidden not in opts, forbidden + " must not be accepted on the command line"

    def test_parser_has_no_anthropic_key_value_option(self):
        """A key passed as --anthropic-key VALUE would land in argv like any other secret."""
        opts = {a for act in build_parser()._actions for a in act.option_strings}
        assert "--anthropic-key" not in opts
        assert "--with-anthropic-key" in opts, "the opt-in flag should exist instead"

    def test_parser_accepts_the_non_secret_arguments(self):
        args = build_parser().parse_args(["--user", "scott", "--account", "111111111"])
        assert args.user == "scott"
        assert args.account == "111111111"


class TestProvision:
    def test_creates_tenant_with_credentials_and_password(self, store):
        provision(store, "scott", "111111111", password="pw-1234", bearer="rh-token")  # pragma: allowlist secret
        c = store.get("scott")
        assert c.account_number == "111111111"
        assert c.robinhood_bearer == "rh-token"
        assert store.verify_password("scott", "pw-1234") is True

    def test_optional_anthropic_key_is_stored(self, store):
        provision(store, "scott", "1", password="p", bearer="b", anthropic_key="sk-ant-x")
        assert store.get("scott").anthropic_api_key == "sk-ant-x"  # pragma: allowlist secret

    def test_anthropic_key_omitted_leaves_none(self, store):
        provision(store, "scott", "1", password="p", bearer="b")
        assert store.get("scott").anthropic_api_key is None

    def test_invalid_user_id_refused(self, store):
        with pytest.raises(Exception):
            provision(store, "../escape", "1", password="p", bearer="b")

    def test_empty_bearer_refused(self, store):
        with pytest.raises(ProvisionError):
            provision(store, "scott", "1", password="p", bearer="")

    def test_empty_password_refused(self, store):
        with pytest.raises(ProvisionError):
            provision(store, "scott", "1", password="", bearer="b")

    def test_empty_account_refused(self, store):
        with pytest.raises(ProvisionError):
            provision(store, "scott", "", password="p", bearer="b")


class TestOverwriteProtection:
    def test_existing_tenant_is_refused_by_default(self, store):
        provision(store, "scott", "1", password="p", bearer="b")
        with pytest.raises(ProvisionError, match="already exists"):
            provision(store, "scott", "1", password="p2", bearer="b2")  # pragma: allowlist secret

    def test_existing_tenant_updated_when_explicitly_allowed(self, store):
        provision(store, "scott", "1", password="p", bearer="b")
        provision(store, "scott", "2", password="p2", bearer="b2", update=True)  # pragma: allowlist secret
        assert store.get("scott").account_number == "2"
        assert store.verify_password("scott", "p2") is True

    def test_refusal_does_not_alter_the_existing_record(self, store):
        provision(store, "scott", "1", password="p", bearer="b")
        with pytest.raises(ProvisionError):
            provision(store, "scott", "999", password="other", bearer="other")  # pragma: allowlist secret
        assert store.get("scott").account_number == "1"
        assert store.verify_password("scott", "p") is True


class TestAccountUniqueness:
    def test_two_tenants_cannot_share_a_brokerage_account(self, store):
        """A shared account number makes the trading guard's ownership check meaningless."""
        provision(store, "scott", "111111111", password="p", bearer="b")
        with pytest.raises(ProvisionError, match="already claimed"):
            provision(store, "darren", "111111111", password="p", bearer="b")

    def test_distinct_accounts_are_fine(self, store):
        provision(store, "scott", "111111111", password="p", bearer="b")
        provision(store, "darren", "222222222", password="p", bearer="b")
        assert set(store.list_user_ids()) == {"scott", "darren"}


class TestStoredSecretsAreEncrypted:
    def test_bearer_and_password_absent_from_the_db_file(self, tmp_path):
        path = tmp_path / "cred.db"
        s = CredentialStore(path, key=generate_key())
        provision(
            s,
            "scott",
            "1",
            password="PLAINTEXT_PW_MARKER",  # pragma: allowlist secret
            bearer="PLAINTEXT_BEARER_MARKER",  # pragma: allowlist secret
        )
        raw = path.read_bytes()
        assert b"PLAINTEXT_PW_MARKER" not in raw
        assert b"PLAINTEXT_BEARER_MARKER" not in raw
        # sanity: the row really is there
        con = sqlite3.connect(path)
        assert con.execute("SELECT count(*) FROM tenant_credentials").fetchone()[0] == 1
        con.close()
